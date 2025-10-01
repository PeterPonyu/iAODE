from pydantic import BaseModel, Field
from typing import Literal


class DataInfo(BaseModel):  

    n_cells: int = Field(...)  
    n_genes: int = Field(...)

class AgentParams(BaseModel):  

    layer: str = Field(default="counts")  
    batch_percent: float = Field(default=0.01)  
    recon: float = Field(default=1.0)  
    irecon: float = Field(default=0.0)  
    beta: float = Field(default=1.0)  
    dip: float = Field(default=0.0)  
    tc: float = Field(default=0.0)  
    info: float = Field(default=0.0)  
    hidden_dim: int = Field(default=128)  
    latent_dim: int = Field(default=10)  
    i_dim: int = Field(default=2)  
    use_ode: bool = Field(default=False)  
    loss_mode: Literal["mse", "nb", "zinb"] = Field(default="nb")  
    lr: float = Field(default=1e-4)  
    vae_reg: float = Field(default=0.5)  
    ode_reg: float = Field(default=0.5)  

class TrainParams(BaseModel):  

    epochs: int = Field(default=3000) 


class TraningState(BaseModel):
    
    epoch: int = Field(...)
    ARI: float = Field(...)
    NMI: float = Field(...)
    ASW: float = Field(...)
    CAL: float = Field(...)
    DAV: float = Field(...)
    COR: float = Field(...)
