from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .model import DataInfo, AgentParams, TrainParams, TraningState
from iaode import scATACAgent as agent
import tempfile
from typing import Literal
from datetime import datetime
import os
import scanpy as sc
import pandas as pd

app = FastAPI(title="iAODE API", version="0.1.0")

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the iAODE API"}


class app_state:
    def __init__(self):
        self.agent = None
        self.adata = None
        self.training_state = {}
        self.current_epoch = 0

state = app_state()

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5ad') as temp_file:

        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
        state.adata = sc.read_h5ad(temp_file_path)

    n_cells, n_genes = state.adata.shape
    return DataInfo(n_cells=n_cells, n_genes=n_genes)

def run_training_loop(trainparams):
    """
    This function now assumes state.agent is already created.
    """
    for e in range(trainparams.epochs):

        batch_data = state.agent.sample_training_batch()
        metrics = state.agent.train_and_evaluate(batch_data)

        state.training_state = metrics
        state.current_epoch = e + 1

    
    state.training_state["status"] = "Finished"
    print("Training finished.")

@app.post("/train")
async def train_model(
    agparams: AgentParams,
    trainparams: TrainParams,
    background_tasks: BackgroundTasks
):

    state.agent = agent(adata=state.adata, **agparams.model_dump())
    
    state.current_epoch = 0
    state.training_state = {"status": "Starting"}

    background_tasks.add_task(run_training_loop, trainparams)

    return {"message": "Train finished"}

@app.get("/state")
async def get_state():
    return TraningState(epoch=state.current_epoch, **state.training_state)


@app.get("/download")
async def get_latent(embedding_type: Literal["latent", "interpretable"] = Query(
        ...
        )):

    if state.agent is None:
        return {"error": "Agent has not been trained yet. Call /train first."}, 400
    
    res = state.agent.get_representations()

    os.makedirs('downloads/', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    latent_filename = f"{embedding_type}_embedding_{timestamp}.csv"
    latent_filepath = os.path.join('downloads/', latent_filename)

    pd.DataFrame(res[embedding_type]).to_csv(latent_filepath)
        
    return FileResponse(
        path=latent_filepath,
        filename=os.path.basename(latent_filepath),
        media_type='text/csv'
    )

