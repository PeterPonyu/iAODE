
from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from .model import DataInfo, AgentParams, TrainParams, TrainingState, TFIDFParams, HVPParams, SubsampleParams, PreprocessInfo
from iaode.utils import tfidf_normalization, select_highly_variable_peaks, subsample_cells_and_peaks
from iaode.agent import agent
from anndata import AnnData
import tempfile
import os
from typing import Literal, Optional
from datetime import datetime
import scanpy as sc
import pandas as pd
from pathlib import Path

VERSION = "0.3.0"

app = FastAPI(title="iAODE API", version=VERSION)

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files from Next.js build
frontend_path = Path(__file__).parent.parent / "frontend" / "out"
if frontend_path.exists():
    # Mount Next.js static assets
    app.mount("/_next", StaticFiles(directory=str(frontend_path / "_next")), name="next-static")
    
    # Serve other static files
    for static_file in ["favicon.ico", "file.svg", "globe.svg", "next.svg", "vercel.svg", "window.svg"]:
        if (frontend_path / static_file).exists():
            @app.get(f"/{static_file}")
            async def serve_static(file=static_file):
                return FileResponse(frontend_path / file)



class AppState:
    """Application state management with proper typing"""
    def __init__(self) -> None:
        self.agent: Optional[agent] = None
        self.adata: Optional[AnnData] = None
        self.status: str = "idle"
        self.current_epoch: int = 0
        self.message: str = ""


state = AppState()


# Serve frontend pages
@app.get("/ui", response_class=HTMLResponse)
@app.get("/ui/", response_class=HTMLResponse)
async def serve_ui_root():
    """Serve the frontend homepage"""
    if frontend_path.exists():
        return FileResponse(frontend_path / "index.html")
    return {"message": "Frontend not built. Run 'cd frontend && npm run build'"}


@app.get("/ui/train", response_class=HTMLResponse)
async def serve_ui_train():
    """Serve the training page"""
    if frontend_path.exists():
        return FileResponse(frontend_path / "train.html")
    return {"message": "Frontend not built. Run 'cd frontend && npm run build'"}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the iAODE API", "version": VERSION}


@app.post("/upload", response_model=DataInfo)
async def upload_data(
    file: UploadFile = File(...),
    data_type: Literal["scrna", "scatac"] = Query(default="scrna", description="Data type: scrna or scatac")
):
    """Upload AnnData file (.h5ad) or 10x H5 file (.h5)"""
    try:
        # Get file extension
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        
        if file_ext not in ['.h5ad', '.h5']:
            raise ValueError("File must be .h5ad or .h5 format")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Read file based on format
        if file_ext == '.h5ad':
            state.adata = sc.read_h5ad(temp_file_path)
        else:  # .h5
            # For scATAC, set gex_only=False to read peaks data
            gex_only = (data_type == "scrna")
            state.adata = sc.read_10x_h5(temp_file_path, gex_only=gex_only)
        
        if 'counts' not in state.adata.layers:
            state.adata.layers['counts'] = state.adata.X.copy()

        os.unlink(temp_file_path)  # Clean up temp file
        
        n_cells, n_genes = state.adata.shape
        state.status = "data_loaded"
        state.message = f"Loaded {n_cells} cells × {n_genes} features ({data_type})"
        
        return DataInfo(n_cells=n_cells, n_genes=n_genes)
    
    except Exception as e:
        state.status = "error"
        state.message = f"Upload failed: {str(e)}"
        raise


def run_training(agparams: AgentParams, trainparams: TrainParams) -> None:
    """Background task for model training"""
    try:
        state.status = "initializing"
        state.message = "Initializing agent..."
        
        # Initialize agent
        if state.adata is None:
            raise ValueError("No data loaded")
            
        state.agent = agent(adata=state.adata, **agparams.model_dump())
        
        state.status = "training"
        state.message = f"Training for up to {trainparams.epochs} epochs..."
        
        # Train using agent.fit() method
        state.agent.fit(
            epochs=trainparams.epochs,
            patience=trainparams.patience,
            val_every=trainparams.val_every,
            early_stop=trainparams.early_stop
        )
        
        state.status = "completed"
        state.message = "Training completed successfully"
        
    except Exception as e:
        state.status = "error"
        state.message = f"Training failed: {str(e)}"
        print(f"Training error: {e}")


@app.post("/train")
async def train_model(
    agparams: AgentParams,
    trainparams: TrainParams,
    background_tasks: BackgroundTasks
):
    """Initialize and train the model"""
    if state.adata is None:
        return {"error": "No data uploaded. Please upload data first."}
    
    if state.status == "training":
        return {"error": "Training already in progress"}
    
    # Reset state
    state.current_epoch = 0
    state.status = "queued"
    state.message = "Training queued"
    
    # Start training in background
    background_tasks.add_task(run_training, agparams, trainparams)
    
    return {
        "message": "Training started",
        "params": {
            "agent": agparams.model_dump(),
            "training": trainparams.model_dump()
        }
    }


@app.get("/state", response_model=TrainingState)
async def get_state():
    """Get current training state"""
    return TrainingState(
        status=state.status,
        current_epoch=state.current_epoch,
        message=state.message
    )


@app.get("/download")
async def download_embedding(
    embedding_type: Literal["latent", "interpretable"] = Query(..., description="Type of embedding to download")
):
    """Download trained embeddings"""
    if state.agent is None:
        return {"error": "Agent not initialized. Train a model first."}
    
    if state.status != "completed":
        return {"error": f"Training not completed. Current status: {state.status}"}
    
    try:
        # Get embeddings using agent methods
        if embedding_type == "latent":
            embedding = state.agent.get_latent()
        else:  # interpretable
            embedding = state.agent.get_iembed()
        
        # Convert to numpy if tensor
        if hasattr(embedding, 'cpu'):
            embedding = embedding.cpu().detach().numpy()
        
        # Create downloads directory
        os.makedirs('downloads', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{embedding_type}_embedding_{timestamp}.csv"
        filepath = os.path.join('downloads', filename)
        
        # Save as CSV
        pd.DataFrame(embedding).to_csv(filepath, index=False)
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='text/csv'
        )
    
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}


@app.delete("/reset")
async def reset_state():
    """Reset application state"""
    state.agent = None
    state.adata = None
    state.status = "idle"
    state.current_epoch = 0
    state.message = ""
    
    return {"message": "State reset successfully"}


@app.post("/preprocess/tfidf", response_model=PreprocessInfo)
async def normalize_tfidf(params: TFIDFParams):
    """Apply TF-IDF normalization to scATAC-seq data"""
    if state.adata is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first.")
    
    try:
        state.status = "processing"
        state.message = "Applying TF-IDF normalization..."
        
        # Apply TF-IDF (modifies in place)
        tfidf_normalization(
            state.adata,
            scale_factor=params.scale_factor,
            log_tf=params.log_tf,
            log_idf=params.log_idf,
            inplace=True
        )
        
        state.status = "data_loaded"
        state.message = f"TF-IDF normalization completed"
        
        return PreprocessInfo(
            n_cells=state.adata.n_obs,
            n_peaks=state.adata.n_vars,
            message=f"TF-IDF applied (scale={params.scale_factor:.0e})"
        )
    
    except ValueError as e:
        state.status = "error"
        state.message = f"TF-IDF failed: {str(e)}"
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        state.status = "error"
        state.message = f"TF-IDF failed: {str(e)}"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/select-hvp", response_model=PreprocessInfo)
async def select_hvp(params: HVPParams):
    """Select highly variable peaks from scATAC-seq data"""
    if state.adata is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first.")
    
    try:
        state.status = "processing"
        state.message = "Selecting highly variable peaks..."
        
        # Select HVP (adds 'highly_variable' column to adata.var)
        select_highly_variable_peaks(
            state.adata,
            n_top_peaks=params.n_top_peaks,
            min_accessibility=params.min_accessibility,
            max_accessibility=params.max_accessibility,
            method=params.method,
            use_raw_counts=params.use_raw_counts,
            inplace=True
        )
        
        n_hvp = state.adata.var['highly_variable'].sum()
        
        state.status = "data_loaded"
        state.message = f"Selected {n_hvp} highly variable peaks"
        
        return PreprocessInfo(
            n_cells=state.adata.n_obs,
            n_peaks=state.adata.n_vars,
            message=f"Selected {n_hvp}/{state.adata.n_vars} peaks using {params.method}"
        )
    
    except ValueError as e:
        state.status = "error"
        state.message = f"HVP selection failed: {str(e)}"
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        state.status = "error"
        state.message = f"HVP selection failed: {str(e)}"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/subsample", response_model=PreprocessInfo)
async def subsample_data(params: SubsampleParams):
    """Subsample cells and optionally filter to highly variable peaks"""
    if state.adata is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first.")
    
    # Validate params
    if params.n_cells is None and params.frac_cells is None:
        raise HTTPException(status_code=400, detail="Must specify either n_cells or frac_cells")
    
    if params.n_cells is not None and params.frac_cells is not None:
        raise HTTPException(status_code=400, detail="Specify only one of n_cells or frac_cells")
    
    try:
        state.status = "processing"
        state.message = "Subsampling data..."
        
        # Subsample (returns new AnnData)
        state.adata = subsample_cells_and_peaks(
            state.adata,
            n_cells=params.n_cells,
            frac_cells=params.frac_cells,
            use_hvp=params.use_hvp,
            hvp_column=params.hvp_column,
            seed=params.seed,
            inplace=False
        )
        
        state.status = "data_loaded"
        state.message = f"Subsampled to {state.adata.n_obs} cells"
        
        return PreprocessInfo(
            n_cells=state.adata.n_obs,
            n_peaks=state.adata.n_vars,
            message=f"Subsampled to {state.adata.n_obs} cells × {state.adata.n_vars} peaks"
        )
    
    except ValueError as e:
        state.status = "error"
        state.message = f"Subsampling failed: {str(e)}"
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        state.status = "error"
        state.message = f"Subsampling failed: {str(e)}"
        raise HTTPException(status_code=500, detail=str(e))
