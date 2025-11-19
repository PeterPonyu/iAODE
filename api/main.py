
from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .model import DataInfo, AgentParams, TrainParams, TrainingState
from iaode.agent import agent
from anndata import AnnData
import tempfile
import os
from typing import Literal, Optional
from datetime import datetime
import scanpy as sc
import pandas as pd

VERSION = "0.2.0"

app = FastAPI(title="iAODE API", version=VERSION)

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    """Application state management with proper typing"""
    def __init__(self) -> None:
        self.agent: Optional[agent] = None
        self.adata: Optional[AnnData] = None
        self.status: str = "idle"
        self.current_epoch: int = 0
        self.message: str = ""


state = AppState()


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
