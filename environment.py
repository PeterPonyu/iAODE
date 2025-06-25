
from .model import iAODEVAE
from .mixin import envMixin
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Any
import torch


class Env(iAODEVAE, envMixin):
    """
    Environment wrapper for iAODEVAE model with integrated evaluation capabilities.
    
    This class combines the iAODEVAE model with environmental evaluation metrics,
    providing a complete training and evaluation framework for single-cell analysis.
    It handles data sampling, model training, and performance tracking through
    various clustering and correlation metrics.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell expression data
    layer : str
        Layer name in adata.layers to use for training data
    percent : float
        Percentage of data to use in each batch (0.0 to 1.0)
    recon : float
        Weight for reconstruction loss
    irecon : float
        Weight for information reconstruction loss
    beta : float
        Weight for KL divergence regularization
    dip : float
        Weight for disentangled information processing loss
    tc : float
        Weight for total correlation loss
    info : float
        Weight for information bottleneck loss (MMD)
    hidden_dim : int
        Dimension of hidden layers in the network
    latent_dim : int
        Dimension of the latent representation space
    i_dim : int
        Dimension of the information bottleneck
    use_ode : bool
        Whether to enable ODE integration for temporal modeling
    loss_mode : str
        Type of loss function ('mse', 'nb', 'zinb')
    lr : float
        Learning rate for the optimizer
    vae_reg : float
        Regularization weight for VAE component
    ode_reg : float
        Regularization weight for ODE component
    device : torch.device
        Device to run computations on
    """
    
    def __init__(
        self,
        adata,
        layer: str,
        percent: float,
        recon: float,
        irecon: float,
        beta: float,
        dip: float,
        tc: float,
        info: float,
        hidden_dim: int,
        latent_dim: int,
        i_dim: int,
        use_ode: bool,
        loss_mode: str,
        lr: float,
        vae_reg: float,
        ode_reg: float,
        device: torch.device,
        *args,
        **kwargs,
    ):
        # Register and preprocess the AnnData object
        self._register_anndata(adata, layer, latent_dim)
        
        # Calculate batch size based on percentage
        self.batch_size = int(percent * self.n_obs)
        
        # Initialize parent iAODEVAE model
        super().__init__(
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            state_dim=self.n_var,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            device=device,
        )
        
        # Initialize score tracking
        self.score = []

    def load_data(self) -> np.ndarray:
        """
        Sample and load training data for the current batch.
        
        This method randomly samples a subset of observations based on the
        configured batch size and returns the corresponding expression data.
        The sampling indices are stored internally for evaluation purposes.
        
        Returns
        -------
        np.ndarray
            Sampled expression data of shape (batch_size, n_genes)
        """
        data, idx = self._sample_data()
        self.idx = idx
        return data

    def step(self, data: np.ndarray) -> None:
        """
        Perform one training step and evaluate model performance.
        
        This method executes a complete training cycle:
        1. Updates the model parameters using the provided data
        2. Extracts latent representations from the data
        3. Computes evaluation scores using various metrics
        4. Stores the scores for tracking progress
        
        Parameters
        ----------
        data : np.ndarray
            Training data batch of shape (batch_size, n_genes)
        """
        # Update model parameters
        self.update(data)
        
        # Extract latent representations
        latent = self.take_latent(data)
        
        # Compute evaluation scores
        score = self._calc_score(latent)
        
        # Store scores for tracking
        self.score.append(score)

    def _sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal method to sample training data from the registered dataset.
        
        This method performs stratified random sampling to select a subset
        of observations for training. It ensures randomization while maintaining
        the ability to track which samples were used for evaluation.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Sampled expression data of shape (batch_size, n_genes)
            - Indices of sampled observations
        """
        # Generate random permutation of all observation indices
        idx = np.random.permutation(self.n_obs)
        
        # Select subset based on batch size
        idx_ = np.random.choice(idx, self.batch_size)
        
        # Extract corresponding data
        data = self.X[idx_, :]
        
        return data, idx_

    def _register_anndata(
        self, 
        adata, 
        layer: str, 
        latent_dim: int
    ) -> None:
        """
        Register and preprocess AnnData object for training.
        
        This method extracts and preprocesses the expression data from the
        specified layer, applies log transformation, and generates reference
        cluster labels for evaluation purposes.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object containing single-cell expression data
        layer : str
            Layer name in adata.layers to use for training data
        latent_dim : int
            Number of clusters to generate for reference labeling
            
        Notes
        -----
        The method applies log1p transformation to handle the typical
        count-based nature of single-cell expression data. Reference
        labels are generated using K-means clustering on the raw data
        for subsequent evaluation of learned representations.
        """
        # Extract and log-transform expression data
        self.X = np.log1p(adata.layers[layer].toarray())
        
        # Store data dimensions
        self.n_obs = adata.shape[0]  # Number of observations (cells)
        self.n_var = adata.shape[1]  # Number of variables (genes)
        
        # Generate reference cluster labels for evaluation
        self.labels = KMeans(latent_dim).fit_predict(self.X)
        
        return

