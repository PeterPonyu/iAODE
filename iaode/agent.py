
import numpy as np
import torch
import scanpy as sc
from anndata import AnnData
from typing import Optional, Literal, Tuple
from scipy.sparse import csr_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
import tqdm

from .environment import scATACEnvironment
from .utils import quiver_autoscale


class scATACAgent(scATACEnvironment):
    """
    Agent for scATAC-seq velocity analysis using iAODEVAE.
    
    Extends scATACEnvironment to provide complete scATAC-seq velocity analysis including
    model fitting, latent representations, data imputation, and velocity field computation.
    
    Parameters
    ----------
    adata : AnnData
        scATAC-seq data with peak accessibility counts
    layer : str, default="counts"
        Data layer to use for training
    batch_percent : float, default=0.01  
        Fraction of data per training batch
    hidden_dim : int, default=128
        Hidden layer dimension
    latent_dim : int, default=10
        Latent space dimension
    i_dim : int, default=2
        Information bottleneck dimension
    use_ode : bool, default=False
        Enable ODE integration for temporal dynamics
    loss_mode : {"mse", "nb", "zinb"}, default="nb"
        Loss function type
    lr : float, default=1e-4
        Learning rate
    device : torch.device, optional
        Computing device
    **kwargs
        Additional parameters for loss weights and regularization
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
        batch_percent: float = 0.01,
        recon: float = 1.0,
        irecon: float = 0.0,
        beta: float = 1.0,
        dip: float = 0.0,
        tc: float = 0.0,
        info: float = 0.0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        use_ode: bool = False,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        lr: float = 1e-4,
        vae_reg: float = 0.5,
        ode_reg: float = 0.5,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        super().__init__(
            adata=adata,
            layer=layer,
            batch_percent=batch_percent,
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            device=device,
            **kwargs
        )

    def fit(self, epochs: int = 1000, verbose: bool = True) -> 'scATACAgent':
        """Train the model with progress monitoring."""
        iterator = tqdm.tqdm(range(epochs), desc="Training", ncols=120) if verbose else range(epochs)
        
        for epoch in iterator:
            batch_data = self.sample_training_batch()
            metrics = self.train_and_evaluate(batch_data)
            
            if verbose and (epoch + 1) % 10 == 0:
                iterator.set_postfix({
                    "Loss": f"{metrics.get('total_loss', 0):.2f}",
                    "Recon": f"{metrics.get('reconstruction', 0):.2f}",
                    "KL": f"{metrics.get('kl_divergence', 0):.2f}",
                    "Sil": f"{metrics.get('latent_silhouette', 0):.2f}"
                })
        
        return self

    def get_representations(self) -> dict:
        """Extract all learned representations."""
        return {
            'latent': self.extract_latent_representations(self.accessibility_matrix),
            'interpretable': self.extract_interpretable_embeddings(self.accessibility_matrix),
            'pseudotime': self.extract_pseudotime(self.accessibility_matrix) if self.use_ode else None
        }

    def get_velocity_gradients(self) -> np.ndarray:
        """Extract velocity gradients from ODE solver."""
        if not self.use_ode:
            raise ValueError("Velocity gradients require ODE mode")
        return self.extract_velocity_gradients(self.accessibility_matrix)

    def impute_data(
        self, 
        top_k: int = 30, 
        alpha: float = 0.9, 
        n_steps: int = 3, 
        decay: float = 0.99
    ) -> np.ndarray:
        """
        Impute data using transition-based smoothing.
        
        Parameters
        ----------
        top_k : int
            Number of top transitions for sparsification
        alpha : float
            Blending weight (0=original, 1=fully imputed)
        n_steps : int
            Number of diffusion steps
        decay : float
            Decay factor for multi-step contributions
        """
        T = self.compute_transition_probabilities(self.accessibility_matrix, top_k)
        
        # Multi-step diffusion
        X_current = self.accessibility_matrix.copy()
        X_imputed = self.accessibility_matrix.copy()
        
        for i in range(n_steps):
            X_current = T @ X_current
            X_imputed += (decay ** i) * X_current
            
        # Normalize and blend
        total_weight = 1 + sum(decay ** i for i in range(n_steps))
        X_imputed = X_imputed / total_weight
        
        return (1 - alpha) * self.accessibility_matrix + alpha * X_imputed

    def compute_velocity_field(
        self,
        adata: AnnData,
        latent_key: str,
        embedding_key: str,
        velocity_key: str = "velocity",
        n_neighbors: int = 20,
        scale_factor: int = 10,
        grid_density: float = 1.0,
        smoothing: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute complete velocity field for visualization.
        
        Parameters
        ----------
        adata : AnnData
            Data object to store results
        latent_key : str
            Key for latent embeddings in adata.obsm
        embedding_key : str  
            Key for 2D embeddings in adata.obsm
        velocity_key : str
            Key to store velocity results
        n_neighbors : int
            Number of neighbors for graph construction
        scale_factor : int
            Scaling factor for transition probabilities
        grid_density : float
            Grid density for visualization
        smoothing : float
            Smoothing parameter for grid interpolation
        """
        if not self.use_ode:
            raise ValueError("Velocity field computation requires ODE mode")
            
        # Store velocity gradients
        gradients = self.get_velocity_gradients()
        adata.obsm[f"{velocity_key}_gradients"] = gradients
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(
            adata, latent_key, f"{velocity_key}_gradients", n_neighbors
        )
        adata.obsp["similarities"] = similarity_matrix
        
        # Derive velocity field
        velocity_field = self._derive_velocity_field(
            adata, "similarities", embedding_key, scale_factor
        )
        adata.obsm[velocity_key] = velocity_field
        
        # Create grid representation
        E = np.array(adata.obsm[embedding_key])
        V = velocity_field
        E_grid, V_grid = self._create_velocity_grid(E, V, grid_density, smoothing)
        
        return E_grid, V_grid

    def _compute_similarity_matrix(
        self,
        adata: AnnData,
        latent_key: str,
        gradient_key: str,
        n_neighbors: int
    ) -> csr_matrix:
        """Compute cosine similarity between embeddings and gradients."""
        
        # Set up neighborhood graph
        sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=n_neighbors)
        
        Z = np.array(adata.obsm[latent_key])
        V = np.array(adata.obsm[gradient_key])
        n_cells = adata.n_obs
        
        # Compute similarities efficiently
        similarities = []
        indices = []
        indptr = [0]
        
        for i in range(n_cells):
            # Get neighbors
            neighbor_idx = adata.obsp["distances"][i].indices
            
            # Compute embedding differences
            dZ = Z[neighbor_idx] - Z[i]
            
            # Compute cosine similarities
            cos_sim = np.einsum("ij,j->i", dZ, V[i])
            cos_sim /= (np.linalg.norm(dZ, axis=1) * np.linalg.norm(V[i]) + 1e-12)
            cos_sim = np.clip(cos_sim, -1, 1)
            cos_sim[np.isnan(cos_sim)] = 0
            
            similarities.extend(cos_sim)
            indices.extend(neighbor_idx)
            indptr.append(len(similarities))
        
        return csr_matrix(
            (similarities, indices, indptr), 
            shape=(n_cells, n_cells)
        )

    def _derive_velocity_field(
        self,
        adata: AnnData,
        similarity_key: str,
        embedding_key: str,
        scale_factor: int
    ) -> np.ndarray:
        """Derive velocity field from similarities and embeddings."""
        
        T = adata.obsp[similarity_key].copy()
        E = np.array(adata.obsm[embedding_key])
        
        # Apply exponential scaling and normalization
        T.data = np.sign(T.data) * np.expm1(np.abs(T.data) * scale_factor)
        T = T.multiply(csr_matrix(1.0 / (np.abs(T).sum(1) + 1e-12)))
        
        # Compute velocity for each cell
        V = np.zeros_like(E)
        for i in range(adata.n_obs):
            if T[i].nnz > 0:
                neighbor_idx = T[i].indices
                weights = T[i].data
                
                # Compute weighted displacement
                dE = E[neighbor_idx] - E[i]
                dE_norm = dE / (np.linalg.norm(dE, axis=1, keepdims=True) + 1e-12)
                
                V[i] = weights @ dE_norm - weights.mean() * dE_norm.sum(0)
        
        # Normalize velocity magnitude
        V /= (3 * quiver_autoscale(E, V) + 1e-12)
        
        return V

    def _create_velocity_grid(
        self,
        embeddings: np.ndarray,
        velocities: np.ndarray,
        density: float,
        smoothing: float,
        streamplot_format: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create grid representation for velocity visualization."""
        
        # Create coordinate grids
        grid_points = int(50 * density)
        grids = []
        
        for dim in range(embeddings.shape[1]):
            min_val, max_val = embeddings[:, dim].min(), embeddings[:, dim].max()
            margin = 0.01 * (max_val - min_val)
            grid = np.linspace(min_val - margin, max_val + margin, grid_points)
            grids.append(grid)
        
        # Create meshgrid
        if streamplot_format and len(grids) == 2:
            E_grid = np.stack(grids)
            mesh_coords = np.stack(np.meshgrid(*grids), axis=-1)
            grid_flat = mesh_coords.reshape(-1, 2)
        else:
            mesh_coords = np.meshgrid(*grids)
            grid_flat = np.column_stack([m.flat for m in mesh_coords])
            E_grid = grid_flat
        
        # Interpolate velocities using k-NN
        n_neighbors = min(embeddings.shape[0] // 50, 50)
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(embeddings)
        
        distances, neighbor_indices = nn.kneighbors(grid_flat)
        
        # Gaussian weighting
        scale = np.mean([g[1] - g[0] for g in grids]) * smoothing
        weights = norm.pdf(distances, scale=scale)
        weight_sums = weights.sum(axis=1)
        
        # Weighted interpolation
        V_grid = np.zeros((grid_flat.shape[0], velocities.shape[1]))
        for i, (neighs, ws) in enumerate(zip(neighbor_indices, weights)):
            if weight_sums[i] > 0:
                V_grid[i] = (velocities[neighs] * ws[:, np.newaxis]).sum(0) / weight_sums[i]
        
        if streamplot_format and len(grids) == 2:
            # Format for matplotlib streamplot
            V_grid = V_grid.T.reshape(2, grid_points, grid_points)
            
            # Apply quality filtering
            magnitude = np.sqrt((V_grid ** 2).sum(0))
            low_quality = magnitude < np.percentile(magnitude, 5)
            V_grid[:, low_quality] = np.nan
        
        return E_grid, V_grid

    def analyze_trajectory(
        self,
        adata: AnnData,
        latent_key: str = "X_latent",
        embedding_key: str = "X_umap", 
        velocity_key: str = "velocity",
        save_results: bool = True
    ) -> dict:
        """
        Complete trajectory analysis pipeline.
        
        Parameters
        ----------
        adata : AnnData
            Data object for analysis
        latent_key : str
            Key for latent representations
        embedding_key : str
            Key for 2D embeddings  
        velocity_key : str
            Key for velocity results
        save_results : bool
            Whether to save results to adata
            
        Returns
        -------
        dict
            Analysis results including representations and velocity field
        """
        # Extract representations
        representations = self.get_representations()
        
        if save_results:
            adata.obsm[latent_key] = representations['latent']
            adata.obsm[f"{latent_key}_interpretable"] = representations['interpretable']
            if representations['pseudotime'] is not None:
                adata.obs['pseudotime'] = representations['pseudotime']
        
        # Compute velocity field
        if self.use_ode:
            E_grid, V_grid = self.compute_velocity_field(
                adata, latent_key, embedding_key, velocity_key
            )
            
            return {
                'representations': representations,
                'velocity_grid': (E_grid, V_grid),
                'imputed_data': self.impute_data() if save_results else None
            }
        else:
            return {
                'representations': representations,
                'imputed_data': self.impute_data() if save_results else None
            }

