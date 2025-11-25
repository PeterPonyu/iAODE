
from pydantic import BaseModel, Field
from typing import Literal, Optional

# ===============================================
# TRAINING PYDANTIC MODELS
# ===============================================

class DataInfo(BaseModel):
    """Information about uploaded dataset"""
    n_cells: int = Field(..., description="Number of cells in the dataset")
    n_genes: int = Field(..., description="Number of genes in the dataset")

class AgentParams(BaseModel):
    """Parameters for agent initialization - matches iaode/agent.py"""
    layer: str = Field(default="counts", description="Layer to use from AnnData")
    recon: float = Field(default=1.0, description="Reconstruction loss weight")
    irecon: float = Field(default=0.0, description="Interpretable reconstruction loss weight")
    beta: float = Field(default=1.0, description="Beta-VAE regularization weight")
    dip: float = Field(default=0.0, description="DIP-VAE regularization weight")
    tc: float = Field(default=0.0, description="Total correlation loss weight")
    info: float = Field(default=0.0, description="Information loss weight")
    hidden_dim: int = Field(default=128, description="Hidden layer dimension")
    latent_dim: int = Field(default=10, description="Latent space dimension")
    i_dim: int = Field(default=2, description="Interpretable embedding dimension")
    use_ode: bool = Field(default=False, description="Whether to use ODE")
    loss_mode: Literal["mse", "nb", "zinb"] = Field(default="nb", description="Loss function type")
    lr: float = Field(default=1e-4, description="Learning rate")
    vae_reg: float = Field(default=0.5, description="VAE regularization weight")
    ode_reg: float = Field(default=0.5, description="ODE regularization weight")
    train_size: float = Field(default=0.7, description="Training set proportion")
    val_size: float = Field(default=0.15, description="Validation set proportion")
    test_size: float = Field(default=0.15, description="Test set proportion")
    batch_size: int = Field(default=128, description="Batch size for training")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


class TrainParams(BaseModel):
    """Parameters for model training - matches agent.fit()"""
    epochs: int = Field(default=100, description="Maximum number of training epochs")
    patience: int = Field(default=20, description="Early stopping patience")
    val_every: int = Field(default=5, description="Validation frequency (epochs)")
    early_stop: bool = Field(default=True, description="Whether to use early stopping")


class TrainingState(BaseModel):
    """Current training state"""
    status: str = Field(..., description="Training status")
    current_epoch: int = Field(default=0, description="Current epoch number")
    message: str = Field(default="", description="Additional information")

class TFIDFParams(BaseModel):
    """Parameters for TF-IDF normalization"""
    scale_factor: float = Field(default=1e4, description="Scale factor (1e4 standard, 1e6 for large datasets)", gt=0)
    log_tf: bool = Field(default=False, description="Log-transform TF")
    log_idf: bool = Field(default=True, description="Log-transform IDF")


class HVPParams(BaseModel):
    """Parameters for highly variable peak selection"""
    n_top_peaks: int = Field(default=20000, description="Number of peaks to select", gt=0)
    min_accessibility: float = Field(default=0.01, description="Min fraction of cells with peak", ge=0.0, le=1.0)
    max_accessibility: float = Field(default=0.95, description="Max fraction of cells", ge=0.0, le=1.0)
    method: Literal['signac', 'snapatac2', 'deviance'] = Field(default='signac', description="Selection method")
    use_raw_counts: bool = Field(default=True, description="Use raw counts for selection")


class SubsampleParams(BaseModel):
    """Parameters for cell and peak subsampling"""
    n_cells: Optional[int] = Field(default=None, description="Number of cells to sample", gt=0)
    frac_cells: Optional[float] = Field(default=None, description="Fraction of cells (0-1)", gt=0.0, le=1.0)
    use_hvp: bool = Field(default=True, description="Filter to highly variable peaks")
    hvp_column: str = Field(default='highly_variable', description="Column name for HVP")
    seed: int = Field(default=42, description="Random seed")


class PreprocessInfo(BaseModel):
    """Information about preprocessing results"""
    n_cells: int = Field(..., description="Number of cells")
    n_peaks: int = Field(..., description="Number of peaks")
    message: str = Field(..., description="Processing message")