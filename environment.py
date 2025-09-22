
import numpy as np
import torch
from typing import Tuple, List, Any, Optional, Union
from sklearn.cluster import KMeans

from scipy import sparse
import warnings
from pathlib import Path

try:
    import scanpy as sc
    from anndata import AnnData
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    AnnData = Any  # Fallback type hint

from .model import iAODEVAE
from .mixin import envMixin


class scATACEnvironment(iAODEVAE, envMixin):
    """
    Comprehensive training and evaluation environment for scATAC-seq analysis using iAODEVAE.
    
    This environment wrapper provides an integrated framework for training the iAODEVAE model
    on single-cell ATAC-seq (chromatin accessibility) data with automatic evaluation metrics,
    batch sampling strategies, and performance tracking capabilities.
    
    Key Features:
    - Flexible batch sampling with stratified and random strategies  
    - Real-time performance evaluation using clustering and correlation metrics
    - Integration with scanpy ecosystem for single-cell analysis
    - Support for sparse matrices and large-scale datasets
    - Comprehensive logging and checkpointing capabilities
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing scATAC-seq accessibility data.
        Expected to have peak accessibility counts in specified layer.
    layer : str
        Layer name in adata.layers containing the accessibility count matrix.
        Common choices: 'X', 'counts', 'raw', 'tfidf'
    batch_percent : float
        Percentage of total cells to use in each training batch (0.0 to 1.0).
        Smaller values provide more stochastic training but faster iterations.
    recon : float
        Weight coefficient for primary reconstruction loss
    irecon : float  
        Weight coefficient for information bottleneck reconstruction loss
    beta : float
        Weight coefficient for KL divergence regularization (β-VAE)
    dip : float
        Weight coefficient for Disentangled Information Processing loss
    tc : float
        Weight coefficient for Total Correlation loss (β-TC-VAE)
    info : float
        Weight coefficient for information bottleneck regularization (MMD)
    hidden_dim : int
        Dimension of hidden layers in encoder/decoder networks
    latent_dim : int
        Dimension of the primary latent representation space
    i_dim : int
        Dimension of the interpretable information bottleneck layer
    use_ode : bool
        Whether to enable Neural ODE integration for temporal dynamics
    loss_mode : Literal["mse", "nb", "zinb"]
        Probabilistic distribution for scATAC-seq data modeling
    lr : float
        Learning rate for Adam optimizer
    vae_reg : float
        Regularization weight for standard VAE latent representations
    ode_reg : float  
        Regularization weight for ODE-integrated latent representations
    device : torch.device
        Computing device (CPU/CUDA)
    reference_clusters : int, optional
        Number of reference clusters for evaluation. If None, uses latent_dim.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.
        
    Attributes
    ----------
    accessibility_matrix : np.ndarray
        Preprocessed scATAC-seq accessibility matrix
    batch_size : int
        Computed batch size based on batch_percent
    reference_labels : np.ndarray
        Reference cluster labels for evaluation
    training_scores : list
        History of evaluation scores during training
    current_batch_indices : np.ndarray
        Indices of cells in current training batch
    
    Examples
    --------
    >>> import scanpy as sc
    >>> import torch
    >>> 
    >>> # Load scATAC-seq data
    >>> adata = sc.read_h5ad("scatac_data.h5ad")
    >>> 
    >>> # Initialize environment
    >>> env = scATACEnvironment(
    ...     adata=adata, layer="counts", batch_percent=0.1,
    ...     recon=1.0, irecon=0.5, beta=1.0, dip=0.1, tc=0.1, info=0.1,
    ...     hidden_dim=512, latent_dim=64, i_dim=16, use_ode=True,
    ...     loss_mode="zinb", lr=1e-3, vae_reg=0.7, ode_reg=0.3,
    ...     device=torch.device("cuda")
    ... )
    >>> 
    >>> # Training loop
    >>> for epoch in range(100):
    ...     batch_data = env.sample_training_batch()
    ...     env.train_and_evaluate(batch_data)
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str,
        batch_percent: float,
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
        reference_clusters: Optional[int] = None,
        random_seed: int = 42,
        *args,
        **kwargs,
    ) -> None:
        
        # Validate inputs
        self._validate_inputs(adata, layer, batch_percent, reference_clusters or latent_dim)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Store configuration
        self.random_seed = random_seed
        
        # Process and register scATAC-seq data
        self._preprocess_scatac_data(adata, layer, reference_clusters or latent_dim)
        
        # Calculate batch size
        self.batch_size = max(1, int(batch_percent * self.n_cells))
        
        # Initialize parent iAODEVAE model
        super().__init__(
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            state_dim=self.n_peaks,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            device=device,
            *args,
            **kwargs
        )
        
        # Initialize evaluation tracking
        self.training_scores = []
        self.current_batch_indices = None
        
        # Log initialization summary
        self._log_initialization_summary()
    
    def _validate_inputs(
        self, adata: AnnData, layer: str, batch_percent: float, n_clusters: int
    ) -> None:
        """Validate input parameters and data compatibility."""
        if not SCANPY_AVAILABLE:
            raise ImportError("scanpy and anndata are required but not installed")
        
        if not hasattr(adata, 'layers') or layer not in adata.layers:
            available_layers = list(adata.layers.keys()) if hasattr(adata, 'layers') else []
            raise ValueError(f"Layer '{layer}' not found. Available layers: {available_layers}")
        
        if not (0.0 < batch_percent <= 1.0):
            raise ValueError(f"batch_percent must be in (0, 1], got {batch_percent}")
        
        if n_clusters <= 1:
            raise ValueError(f"Number of clusters must be > 1, got {n_clusters}")
        
        if adata.shape[0] < 10:
            raise ValueError(f"Dataset too small for training: {adata.shape[0]} cells")
    
    def _preprocess_scatac_data(
        self, adata: AnnData, layer: str, n_reference_clusters: int
    ) -> None:
        """
        Preprocess scATAC-seq data.
        
        Parameters
        ----------
        adata : AnnData
            Input annotated data object
        layer : str
            Data layer to extract
        n_reference_clusters : int
            Number of reference clusters for evaluation
        """
        # Extract accessibility data
        if sparse.issparse(adata.layers[layer]):
            accessibility_data = adata.layers[layer].toarray()
        else:
            accessibility_data = adata.layers[layer].copy()
        
        # Store original data info
        original_shape = accessibility_data.shape
        
        # Log transformation for count data stabilization
        self.accessibility_matrix = np.log1p(accessibility_data)
        
        # Store dimensions
        self.n_cells, self.n_peaks = self.accessibility_matrix.shape
        
        # Generate reference clusters for evaluation
        self.reference_labels = self._generate_reference_clusters(n_reference_clusters)
        
        # Log preprocessing summary
        print(f"Data preprocessing complete:")
        print(f"  Original shape: {original_shape}")
        print(f"  Final shape: {self.accessibility_matrix.shape}")
        print(f"  Generated {n_reference_clusters} reference clusters")
    
    def _generate_reference_clusters(self, n_clusters: int) -> np.ndarray:
        """
        Generate reference cluster labels for evaluation metrics.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters to generate
            
        Returns
        -------
        np.ndarray
            Cluster labels for each cell
        """
        try:
            # Use subset of data for faster clustering if dataset is large
            if self.n_cells > 10000:
                subset_size = min(5000, self.n_cells)
                subset_indices = np.random.choice(self.n_cells, subset_size, replace=False)
                subset_data = self.accessibility_matrix[subset_indices, :]
                
                # Cluster subset
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
                subset_labels = kmeans.fit_predict(subset_data)
                
                # Assign labels to full dataset
                full_labels = kmeans.predict(self.accessibility_matrix)
            else:
                # Direct clustering for smaller datasets
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
                full_labels = kmeans.fit_predict(self.accessibility_matrix)
            
            return full_labels
        except Exception as e:
            warnings.warn(f"Reference clustering failed: {e}. Using random labels.")
            return np.random.randint(0, n_clusters, self.n_cells)
    
    def sample_training_batch(self, strategy: str = "random") -> np.ndarray:
        """
        Sample a batch of cells for training with different sampling strategies.
        
        Parameters
        ----------
        strategy : str, optional
            Sampling strategy: "random", "stratified", or "balanced". Default is "random".
            
        Returns
        -------
        np.ndarray
            Batch of accessibility data of shape (batch_size, n_peaks)
            
        Notes
        -----
        - "random": Uniform random sampling
        - "stratified": Sample proportionally from reference clusters  
        - "balanced": Equal sampling from each reference cluster
        """
        if strategy == "stratified":
            batch_indices = self._stratified_sampling()
        elif strategy == "balanced":
            batch_indices = self._balanced_sampling()
        else:  # random
            batch_indices = self._random_sampling()
        
        self.current_batch_indices = batch_indices
        return self.accessibility_matrix[batch_indices, :]
    
    def _random_sampling(self) -> np.ndarray:
        """Random uniform sampling of cells."""
        return np.random.choice(self.n_cells, size=self.batch_size, replace=False)
    
    def _stratified_sampling(self) -> np.ndarray:
        """Stratified sampling proportional to cluster sizes."""
        unique_labels, label_counts = np.unique(self.reference_labels, return_counts=True)
        proportions = label_counts / len(self.reference_labels)
        
        batch_indices = []
        for label, prop in zip(unique_labels, proportions):
            n_samples = max(1, int(prop * self.batch_size))
            label_indices = np.where(self.reference_labels == label)[0]
            
            if len(label_indices) >= n_samples:
                sampled = np.random.choice(label_indices, n_samples, replace=False)
            else:
                sampled = np.random.choice(label_indices, n_samples, replace=True)
            
            batch_indices.extend(sampled)
        
        # Adjust to exact batch size
        if len(batch_indices) > self.batch_size:
            batch_indices = np.random.choice(batch_indices, self.batch_size, replace=False)
        elif len(batch_indices) < self.batch_size:
            remaining = self.batch_size - len(batch_indices)
            extra_indices = np.random.choice(self.n_cells, remaining, replace=False)
            batch_indices.extend(extra_indices)
        
        return np.array(batch_indices)
    
    def _balanced_sampling(self) -> np.ndarray:
        """Balanced sampling with equal representation from each cluster."""
        unique_labels = np.unique(self.reference_labels)
        samples_per_cluster = self.batch_size // len(unique_labels)
        
        batch_indices = []
        for label in unique_labels:
            label_indices = np.where(self.reference_labels == label)[0]
            
            if len(label_indices) >= samples_per_cluster:
                sampled = np.random.choice(label_indices, samples_per_cluster, replace=False)
            else:
                sampled = np.random.choice(label_indices, samples_per_cluster, replace=True)
            
            batch_indices.extend(sampled)
        
        # Handle remainder
        remaining = self.batch_size - len(batch_indices)
        if remaining > 0:
            extra_indices = np.random.choice(self.n_cells, remaining, replace=False)
            batch_indices.extend(extra_indices)
        
        return np.array(batch_indices[:self.batch_size])
    
    def train_and_evaluate(
        self, batch_data: np.ndarray, compute_full_metrics: bool = False
    ) -> dict:
        """
        Perform one training step and evaluate model performance.
        
        Parameters
        ----------
        batch_data : np.ndarray
            Training batch of shape (batch_size, n_peaks)
        compute_full_metrics : bool, optional
            Whether to compute metrics on full dataset. Default is False (batch only).
            
        Returns
        -------
        dict
            Dictionary containing training losses and evaluation metrics
        """
        # Perform training step
        training_losses = self.train_step(batch_data)
        
        # Extract learned representations
        if compute_full_metrics:
            # Evaluate on full dataset (expensive but comprehensive)
            latent_repr = self.extract_latent_representations(self.accessibility_matrix)
            interpretable_repr = self.extract_interpretable_embeddings(self.accessibility_matrix)
            evaluation_labels = self.reference_labels
        else:
            # Evaluate on current batch only (faster)
            latent_repr = self.extract_latent_representations(batch_data)
            interpretable_repr = self.extract_interpretable_embeddings(batch_data)
            evaluation_labels = self.reference_labels[self.current_batch_indices]
        
        # Compute evaluation metrics
        evaluation_scores = self._compute_evaluation_metrics(
            latent_repr, interpretable_repr, evaluation_labels
        )
        
        # Combine results
        step_results = {**training_losses, **evaluation_scores}
        
        # Store for history tracking
        self.training_scores.append(step_results)
        
        return step_results
    
    def _compute_evaluation_metrics(
        self, latent_repr: np.ndarray, interpretable_repr: np.ndarray, labels: np.ndarray
    ) -> dict:
        """
        Compute comprehensive evaluation metrics for learned representations.
        
        Parameters
        ----------
        latent_repr : np.ndarray
            Latent representations of shape (n_cells, latent_dim)
        interpretable_repr : np.ndarray
            Interpretable embeddings of shape (n_cells, i_dim)
        labels : np.ndarray
            Reference cluster labels
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        try:
            # Compute clustering and correlation scores from envMixin
            latent_scores = self._calc_score(latent_repr, labels)
            interpretable_scores = self._calc_score(interpretable_repr, labels)
            
            return {
                'latent_silhouette': latent_scores.get('silhouette_score', 0.0),
                'latent_ari': latent_scores.get('adjusted_rand_index', 0.0),
                'latent_nmi': latent_scores.get('normalized_mutual_info', 0.0),
                'interpretable_silhouette': interpretable_scores.get('silhouette_score', 0.0),
                'interpretable_ari': interpretable_scores.get('adjusted_rand_index', 0.0),
                'interpretable_nmi': interpretable_scores.get('normalized_mutual_info', 0.0),
            }
        except Exception as e:
            warnings.warn(f"Evaluation metrics computation failed: {e}")
            return {
                'latent_silhouette': 0.0, 'latent_ari': 0.0, 'latent_nmi': 0.0,
                'interpretable_silhouette': 0.0, 'interpretable_ari': 0.0, 'interpretable_nmi': 0.0,
            }
    
    def get_training_summary(self) -> dict:
        """
        Get comprehensive training summary and statistics.
        
        Returns
        -------
        dict
            Training summary including data info, model config, and performance
        """
        if not self.training_scores:
            return {"message": "No training steps completed yet"}
        
        # Extract metric histories
        losses = {key: [step[key] for step in self.training_scores if key in step] 
                 for key in ['total_loss', 'reconstruction', 'kl_divergence']}
        
        scores = {key: [step[key] for step in self.training_scores if key in step]
                 for key in ['latent_silhouette', 'interpretable_silhouette']}
        
        return {
            'data_info': {
                'n_cells': self.n_cells,
                'n_peaks': self.n_peaks,
                'batch_size': self.batch_size,
            },
            'model_config': {
                'latent_dim': self.nn.encoder.latent_params.out_features // 2,
                'use_ode': self.use_ode,
                'loss_mode': self.loss_mode,
            },
            'training_progress': {
                'total_steps': len(self.training_scores),
                'final_loss': losses['total_loss'][-1] if losses['total_loss'] else None,
                'best_silhouette': max(scores['latent_silhouette']) if scores['latent_silhouette'] else None,
            },
            'loss_trends': {key: {'mean': np.mean(vals), 'std': np.std(vals), 'trend': vals[-10:]} 
                           for key, vals in losses.items() if vals},
            'score_trends': {key: {'mean': np.mean(vals), 'std': np.std(vals), 'trend': vals[-10:]}
                            for key, vals in scores.items() if vals}
        }
    
    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """
        Save model checkpoint and training state.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to save checkpoint file
        """
        checkpoint = {
            'model_state_dict': self.nn.state_dict(),
            'optimizer_state_dict': self.nn_optimizer.state_dict(),
            'training_scores': self.training_scores,
            'config': self.get_training_summary()['model_config'],
            'random_seed': self.random_seed,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Union[str, Path]) -> None:
        """
        Load model checkpoint and restore training state.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nn.load_state_dict(checkpoint['model_state_dict'])
        self.nn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_scores = checkpoint.get('training_scores', [])
        print(f"Checkpoint loaded from {filepath}")
    
    def _log_initialization_summary(self) -> None:
        """Log model initialization summary."""
        print(f"\n{'='*60}")
        print(f"scATAC Environment Initialized")
        print(f"{'='*60}")
        print(f"Data: {self.n_cells} cells × {self.n_peaks} peaks")
        print(f"Batch size: {self.batch_size} ({self.batch_size/self.n_cells*100:.1f}%)")
        print(f"Model: {'ODE' if self.use_ode else 'Standard'} VAE, {self.loss_mode.upper()} loss")
        print(f"Architecture: {self.n_peaks} → {self.nn.encoder.base_network[0].out_features} → {self.nn.encoder.latent_params.out_features//2} → {self.nn.latent_encoder.out_features}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

    # Legacy API compatibility methods
    def load_data(self) -> np.ndarray:
        """Legacy method - use sample_training_batch() instead."""
        return self.sample_training_batch()
    
    def step(self, data: np.ndarray) -> None:
        """Legacy method - use train_and_evaluate() instead."""
        self.train_and_evaluate(data)

