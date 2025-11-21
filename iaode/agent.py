# agent.py

from .environment import Env
import scanpy as sc  # type: ignore
from anndata import AnnData  # type: ignore
import numpy as np
import torch
import tqdm  # type: ignore
import time
from typing import Literal, Optional
from scipy.stats import norm  # type: ignore
from scipy.sparse import issparse, csr_matrix, coo_matrix  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore


class agent(Env):
    """
    iAODE: interpretable Accessibility ODE VAE for scATAC-seq analysis.
    
    Main entry point for training and downstream analysis. Provides methods for:
    - Model training with early stopping
    - Latent representation extraction
    - Trajectory inference (pseudotime & velocity)
    - Vector field visualization
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing scATAC-seq counts
    layer : str, default='counts'
        Layer name containing count data
    
    Loss Weights
    ------------
    recon : float, default=1.0
        Reconstruction loss weight
    irecon : float, default=0.0
        Interpretable bottleneck reconstruction weight (0=disabled)
    beta : float, default=1.0
        KL divergence weight (β-VAE)
    dip : float, default=0.0
        Disentangled Inferred Prior weight (0=disabled)
    tc : float, default=0.0
        Total Correlation weight (β-TC-VAE, 0=disabled)
    info : float, default=0.0
        MMD weight (InfoVAE, 0=disabled)
    
    Architecture
    ------------
    hidden_dim : int, default=128
        Hidden layer dimension
    latent_dim : int, default=10
        Full latent space dimension
    i_dim : int, default=2
        Interpretable bottleneck dimension
    encoder_type : {'mlp', 'mlp_residual', 'linear', 'transformer'}, default='mlp'
        Encoder architecture type
    encoder_num_layers : int, default=2
        Number of encoder layers
    encoder_n_heads : int, default=4
        Number of attention heads (transformer only)
    encoder_d_model : int, optional
        Transformer model dimension (defaults to hidden_dim)
    
    Training
    --------
    loss_mode : {'mse', 'nb', 'zinb'}, default='nb'
        Reconstruction loss type (nb recommended for scATAC-seq)
    lr : float, default=1e-4
        Learning rate
    train_size : float, default=0.7
        Training set fraction
    val_size : float, default=0.15
        Validation set fraction
    test_size : float, default=0.15
        Test set fraction
    batch_size : int, default=128
        Batch size
    random_seed : int, default=42
        Random seed for reproducibility
    
    Trajectory Inference
    -------------------
    use_ode : bool, default=False
        Enable Neural ODE for trajectory inference
    vae_reg : float, default=0.5
        VAE latent weight (for blending with ODE)
    ode_reg : float, default=0.5
        ODE latent weight (for blending with VAE)
    
    device : torch.device
        Computation device (auto-detected)
    
    Examples
    --------
    >>> # Basic training
    >>> ag = agent(adata, latent_dim=10, use_ode=True)
    >>> ag.fit(epochs=200, patience=20)
    >>> latent = ag.get_latent()
    >>> pseudotime = ag.get_pseudotime()
    
    >>> # Vector field visualization
    >>> adata.obsm['X_latent'] = ag.get_latent()
    >>> sc.tl.umap(adata, n_components=2)
    >>> E_grid, V_grid = ag.get_vfres(adata, zs_key='X_latent', E_key='X_umap')
    >>> ax = sc.pl.embedding(adata, color='pseudotime', basis='umap', show=False)
    >>> ax.streamplot(E_grid[0], E_grid[1], V_grid[0], V_grid[1])
    """

    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
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
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        batch_size: int = 128,
        random_seed: int = 42,
        encoder_type: Literal["mlp", "mlp_residual", "linear", "transformer"] = "mlp",
        encoder_num_layers: int = 2,
        encoder_n_heads: int = 4,
        encoder_d_model: Optional[int] = None,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        super().__init__(
            adata=adata,
            layer=layer,
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
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            random_seed=random_seed,
            device=device,
            encoder_type=encoder_type,
            encoder_num_layers=encoder_num_layers,
            encoder_n_heads=encoder_n_heads,
            encoder_d_model=encoder_d_model,
        )

        # Resource tracking
        self.train_time = 0.0
        self.peak_memory_gb = 0.0
        self.actual_epochs = 0

    # ========================================================================
    # Training
    # ========================================================================

    def fit(
        self,
        epochs: int = 100,
        patience: int = 20,
        val_every: int = 5,
        early_stop: bool = True,
    ):
        """Train model with early stopping."""
        
        # Initialize resource tracking
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        # Main progress bar (compact)
        with tqdm.tqdm(total=epochs, desc="Training", position=0) as pbar:
            # Secondary info bar (detailed metrics)
            with tqdm.tqdm(total=0, position=1, bar_format='{desc}') as pbar_info:
                
                for epoch in range(epochs):
                    
                    train_loss = self.train_epoch()

                    if (epoch + 1) % val_every == 0 or epoch == 0:
                        val_loss, val_score = self.validate()
                        val_loss = float(val_loss)
                        
                        if early_stop:
                            should_stop, improved = self.check_early_stopping(val_loss, patience)

                            # Main bar: key metrics only
                            pbar.set_postfix({
                                "trn": f"{train_loss:.2f}",
                                "val": f"{val_loss:.2f}",
                                "pat": f"{self.patience_counter}/{patience}",
                                "↑" if improved else "↓": ""
                            })
                            
                            # Info bar: all metrics
                            pbar_info.set_description_str(
                                f"Metrics | ARI:{val_score[0]:.3f} NMI:{val_score[1]:.3f} "
                                f"ASW:{val_score[2]:.3f} CAL:{val_score[3]:.3f} "
                                f"DAV:{val_score[4]:.3f} COR:{val_score[5]:.3f} | "
                                f"Best:{self.best_val_loss:.3f}"
                            )

                            if should_stop:
                                self.actual_epochs = epoch + 1
                                pbar.write(f"\nEarly stopping at epoch {epoch + 1}")
                                break
                        else:
                            pbar.set_postfix({
                                "trn": f"{train_loss:.2f}",
                                "val": f"{val_loss:.2f}",
                            })
                            pbar_info.set_description_str(
                                f"Metrics | ARI:{val_score[0]:.3f} NMI:{val_score[1]:.3f} "
                                f"ASW:{val_score[2]:.3f} CAL:{val_score[3]:.3f} "
                                f"DAV:{val_score[4]:.3f} COR:{val_score[5]:.3f}"
                            )
                    
                    pbar.update(1)
                else:
                    self.actual_epochs = epochs

        # Record resource usage
        self.train_time = time.time() - start_time
        self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0

        print(f"\n{'='*70}")
        print("Training Complete")
        print(f"{'='*70}")
        print(f"  Epochs: {self.actual_epochs}/{epochs}")
        print(f"  Time: {self.train_time:.2f}s ({self.train_time/self.actual_epochs:.2f}s/epoch)")
        print(f"  Peak GPU Memory: {self.peak_memory_gb:.3f} GB")
        if early_stop:
            print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")

        return self
    # ========================================================================
    # Representation Extraction
    # ========================================================================

    def get_latent(self):
        """
        Extract full latent representation.
        
        Returns
        -------
        latent : np.ndarray
            Latent representations, shape (n_cells, latent_dim)
        """
        return self.take_latent(torch.FloatTensor(self.X).to(self.device))

    def get_iembed(self):
        """
        Extract interpretable bottleneck embedding.
        
        Returns
        -------
        iembed : np.ndarray
            Interpretable embeddings, shape (n_cells, i_dim)
        """
        return self.take_iembed(torch.FloatTensor(self.X).to(self.device))

    def get_test_latent(self):
        """
        Extract latent representation from test set.
        
        Returns
        -------
        latent : np.ndarray
            Test set latents, shape (n_test_cells, latent_dim)
        """
        return self.take_latent(torch.FloatTensor(self.X_test).to(self.device))

    # ========================================================================
    # Trajectory Inference (ODE mode only)
    # ========================================================================

    def get_pseudotime(self):
        """
        Infer pseudotime (requires use_ode=True).
        
        Returns
        -------
        pseudotime : np.ndarray
            Inferred pseudotime values, shape (n_cells,)
        """
        if not self.use_ode:
            raise ValueError(
                "Pseudotime requires use_ode=True. "
                "Reinitialize with: agent(adata, use_ode=True)"
            )
        return self.take_time(self.X)

    def get_velocity(self):
        """
        Compute velocity vectors in latent space (requires use_ode=True).
        
        Returns
        -------
        velocity : np.ndarray
            Velocity vectors, shape (n_cells, latent_dim)
        """
        if not self.use_ode:
            raise ValueError(
                "Velocity requires use_ode=True. "
                "Reinitialize with: agent(adata, use_ode=True)"
            )
        return self.take_grad(self.X)

    def get_transition_matrix(self, top_k: int = 30):
        """
        Compute cell-cell transition probability matrix (requires use_ode=True).
        
        Parameters
        ----------
        top_k : int, default=30
            Number of nearest neighbors to retain
        
        Returns
        -------
        transition : np.ndarray
            Sparse transition probability matrix, shape (n_cells, n_cells)
        """
        if not self.use_ode:
            raise ValueError(
                "Transition matrix requires use_ode=True. "
                "Reinitialize with: agent(adata, use_ode=True)"
            )
        return self.take_transition(self.X, top_k=top_k)

    # ========================================================================
    # Vector Field Analysis
    # ========================================================================

    def get_vfres(
        self,
        adata: AnnData,
        zs_key: str,
        E_key: str,
        vf_key: str = "X_vf",
        T_key: str = "cosine_similarity",
        dv_key: str = "X_dv",
        reverse: bool = False,
        run_neigh: bool = True,
        use_rep_neigh: Optional[str] = None,
        t_key: Optional[str] = None,
        n_neigh: int = 20,
        var_stabilize_transform: bool = False,
        scale: int = 10,
        self_transition: bool = False,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ):
        """
        Compute vector field for visualization (requires use_ode=True).
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        zs_key : str
            Key in adata.obsm for latent space (e.g., 'X_latent')
        E_key : str
            Key in adata.obsm for embedding space (e.g., 'X_umap')
        vf_key : str, default='X_vf'
            Key to store velocity field in adata.obsm
        T_key : str, default='cosine_similarity'
            Key to store transition matrix in adata.obsp
        dv_key : str, default='X_dv'
            Key to store projected velocities in adata.obsm
        reverse : bool, default=False
            Reverse velocity direction
        run_neigh : bool, default=True
            Recompute neighborhood graph
        use_rep_neigh : str, optional
            Representation for neighbor detection (defaults to zs_key)
        t_key : str, optional
            Key in adata.obs for pseudotime constraint
        n_neigh : int, default=20
            Number of neighbors
        var_stabilize_transform : bool, default=False
            Apply variance stabilizing transform
        scale : int, default=10
            Scaling factor for transition probabilities
        self_transition : bool, default=False
            Include self-transitions
        smooth : float, default=0.5
            Smoothing factor for grid interpolation
        stream : bool, default=True
            Return streamplot format (True) or quiver format (False)
        density : float, default=1.0
            Grid density for interpolation
        
        Returns
        -------
        E_grid : np.ndarray
            Grid coordinates for plotting
        V_grid : np.ndarray
            Interpolated velocities on grid
        
        Examples
        --------
        >>> adata.obsm['X_latent'] = ag.get_latent()
        >>> sc.tl.umap(adata)
        >>> E_grid, V_grid = ag.get_vfres(adata, zs_key='X_latent', E_key='X_umap')
        >>> ax = sc.pl.embedding(adata, basis='umap', show=False)
        >>> ax.streamplot(E_grid[0], E_grid[1], V_grid[0], V_grid[1])
        """
        if not self.use_ode:
            raise ValueError(
                "Vector field analysis requires use_ode=True. "
                "Reinitialize with: agent(adata, use_ode=True)"
            )
        
        # Compute velocity gradients
        grads = self.take_grad(self.X)
        adata.obsm[vf_key] = grads
        
        # Compute transition similarity matrix
        adata.obsp[T_key] = self.get_similarity(
            adata,
            zs_key=zs_key,
            vf_key=vf_key,
            reverse=reverse,
            run_neigh=run_neigh,
            use_rep_neigh=use_rep_neigh,
            t_key=t_key,
            n_neigh=n_neigh,
            var_stabilize_transform=var_stabilize_transform,
        )
        
        # Project velocities to embedding space
        adata.obsm[dv_key] = self.get_vf(
            adata,
            T_key=T_key,
            E_key=E_key,
            scale=scale,
            self_transition=self_transition,
        )
        
        # Interpolate onto regular grid
        E = np.asarray(adata.obsm[E_key])
        V = np.asarray(adata.obsm[dv_key])
        E_grid, V_grid = self.get_vfgrid(
            E=E,
            V=V,
            smooth=smooth,
            stream=stream,
            density=density,
        )
        
        return E_grid, V_grid

    def get_similarity(
        self,
        adata: AnnData,
        zs_key: str,
        vf_key: str = "X_vf",
        reverse: bool = False,
        run_neigh: bool = True,
        use_rep_neigh: Optional[str] = None,
        t_key: Optional[str] = None,
        n_neigh: int = 20,
        var_stabilize_transform: bool = False,
    ):
        """
        Compute cosine similarity-based transition matrix.
        
        Computes directed similarities between cells based on velocity alignment.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        zs_key : str
            Key in adata.obsm for latent space
        vf_key : str, default='X_vf'
            Key in adata.obsm for velocity field
        reverse : bool, default=False
            Reverse velocity direction
        run_neigh : bool, default=True
            Recompute neighborhood graph
        use_rep_neigh : str, optional
            Representation for neighbor detection
        t_key : str, optional
            Key in adata.obs for pseudotime constraint
        n_neigh : int, default=20
            Number of neighbors
        var_stabilize_transform : bool, default=False
            Apply sqrt transform for variance stabilization
        
        Returns
        -------
        similarity : scipy.sparse.csr_matrix
            Cosine similarity matrix, shape (n_cells, n_cells)
        """
        Z = np.array(adata.obsm[zs_key])
        V = np.array(adata.obsm[vf_key])
        
        if reverse:
            V = -V
        if var_stabilize_transform:
            V = np.sqrt(np.abs(V)) * np.sign(V)

        ncells = adata.n_obs

        # Build neighborhood graph
        if run_neigh or ("neighbors" not in adata.uns):
            if use_rep_neigh is None:
                use_rep_neigh = zs_key
            elif use_rep_neigh not in adata.obsm:
                raise KeyError(
                    f"`{use_rep_neigh}` not found in `.obsm`. "
                    "Please provide valid `use_rep_neigh`."
                )
            sc.pp.neighbors(adata, use_rep=use_rep_neigh, n_neighbors=n_neigh)
        
        n_neigh = adata.uns["neighbors"]["params"]["n_neighbors"] - 1

        # Pseudotime-constrained neighbors
        if t_key is not None:
            if t_key not in adata.obs:
                raise KeyError(f"`{t_key}` not found in `.obs`.")
            ts = adata.obs[t_key].values
            indices_matrix2 = np.zeros((ncells, n_neigh), dtype=int)
            for i in range(ncells):
                idx = np.abs(ts - ts[i]).argsort()[: (n_neigh + 1)]
                idx = np.setdiff1d(idx, i) if i in idx else idx[:-1]
                indices_matrix2[i] = idx

        # Compute cosine similarities
        vals: list = []
        rows: list = []
        cols: list = []
        for i in range(ncells):
            # Get neighbors (first-order + second-order)
            dist_mat = adata.obsp["distances"]
            row1 = dist_mat[i]
            idx = row1.indices if hasattr(row1, "indices") else np.where(row1 > 0)[0]
            idx2_list = []
            for j in idx:
                r = dist_mat[j]
                if hasattr(r, "indices"):
                    idx2_list.append(r.indices)  # type: ignore[attr-defined]
                else:
                    idx2_list.append(np.where(r > 0)[0])
            idx2 = np.unique(np.concatenate(idx2_list)) if idx2_list else np.array([], dtype=int)
            idx2 = np.setdiff1d(idx2, i)
            
            if t_key is None:
                idx = np.unique(np.concatenate([idx, idx2]))
            else:
                idx = np.unique(np.concatenate([idx, idx2, indices_matrix2[i]]))
            
            # Compute displacement vectors
            dZ = Z[idx] - Z[i, None]
            if var_stabilize_transform:
                dZ = np.sqrt(np.abs(dZ)) * np.sign(dZ)
            
            # Cosine similarity
            cos_sim = np.einsum("ij, j", dZ, V[i]) / (
                l2_norm(dZ, axis=1) * l2_norm(V[i])
            )
            cos_sim[np.isnan(cos_sim)] = 0
            
            vals.extend(cos_sim)
            rows.extend(np.repeat(i, len(idx)))
            cols.extend(idx)

        # Build sparse matrix
        res = coo_matrix((vals, (rows, cols)), shape=(ncells, ncells))
        res.data = np.clip(res.data, -1, 1)
        
        return res.tocsr()

    def get_vf(
        self,
        adata: AnnData,
        T_key: str,
        E_key: str,
        scale: int = 10,
        self_transition: bool = False,
    ):
        """
        Project velocity field onto embedding space.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        T_key : str
            Key in adata.obsp for transition matrix
        E_key : str
            Key in adata.obsm for embedding coordinates
        scale : int, default=10
            Exponential scaling factor for transitions
        self_transition : bool, default=False
            Include self-transitions
        
        Returns
        -------
        V : np.ndarray
            Projected velocity field, shape (n_cells, n_dims)
        """
        T = adata.obsp[T_key].copy()

        if self_transition:
            max_t = T.max(1).A.flatten() if hasattr(T.max(1), 'A') else np.array(T.max(1)).flatten()
            ub = np.percentile(max_t, 98)
            self_t = np.clip(ub - max_t, 0, 1)
            if hasattr(T, 'setdiag'):
                T.setdiag(self_t)  # type: ignore[attr-defined]

        # Apply exponential transform and normalize
        # Exponential weighting with safe sparse/dense handling
        sign_T = T.sign() if hasattr(T, 'sign') else np.sign(T)  # type: ignore[attr-defined]
        if hasattr(sign_T, 'multiply'):
            T = sign_T.multiply(np.expm1(abs(T * scale)))  # type: ignore[attr-defined]
        else:
            T = sign_T * np.expm1(abs(T * scale))
        if hasattr(T, 'multiply'):
            denom = abs(T).sum(1)
            denom = np.maximum(denom, 1e-12)
            T = T.multiply(csr_matrix(1.0 / denom))  # type: ignore[attr-defined]
        else:
            denom = np.maximum(np.abs(T).sum(1, keepdims=True), 1e-12)
            T = T / denom
        
        if self_transition and hasattr(T, 'setdiag'):
            T.setdiag(0)  # type: ignore[attr-defined]
            if hasattr(T, 'eliminate_zeros'):
                T.eliminate_zeros()  # type: ignore[attr-defined]

        # Project to embedding space
        E = np.array(adata.obsm[E_key])
        V = np.zeros(E.shape)

        for i in range(adata.n_obs):
            idx = T[i].indices  # type: ignore[attr-defined]
            dE = E[idx] - E[i, None]
            dE /= l2_norm(dE)[:, None]
            dE[np.isnan(dE)] = 0
            prob = T[i].data
            V[i] = prob.dot(dE) - prob.mean() * dE.sum(0)

        V /= 3 * quiver_autoscale(E, V)
        
        return V

    def get_vfgrid(
        self,
        E: np.ndarray,
        V: np.ndarray,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ):
        """
        Interpolate vector field onto regular grid.
        
        Parameters
        ----------
        E : np.ndarray
            Embedding coordinates, shape (n_cells, n_dims)
        V : np.ndarray
            Velocity vectors, shape (n_cells, n_dims)
        smooth : float, default=0.5
            Gaussian kernel smoothing bandwidth
        stream : bool, default=True
            Format for streamplot (True) or quiver (False)
        density : float, default=1.0
            Grid density multiplier
        
        Returns
        -------
        E_grid : np.ndarray
            Grid coordinates
        V_grid : np.ndarray
            Interpolated velocities
        """
        # Create regular grid
        grs = []
        for i in range(E.shape[1]):
            m, M = np.min(E[:, i]), np.max(E[:, i])
            diff = M - m
            m = m - 0.01 * diff
            M = M + 0.01 * diff
            gr = np.linspace(m, M, int(50 * density))
            grs.append(gr)

        meshes = np.meshgrid(*grs)
        E_grid = np.vstack([i.flat for i in meshes]).T

        # Find neighbors for each grid point
        n_neigh = int(E.shape[0] / 50)
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1)
        nn.fit(E)
        dists, neighs = nn.kneighbors(E_grid)

        # Gaussian kernel smoothing
        scale = np.mean([g[1] - g[0] for g in grs]) * smooth
        weight = norm.pdf(x=dists, scale=scale)
        weight_sum = weight.sum(1)

        V_grid = (V[neighs] * weight[:, :, None]).sum(1)
        V_grid /= np.maximum(1, weight_sum)[:, None]

        if stream:
            # Format for streamplot
            E_grid = np.stack(grs)
            ns = E_grid.shape[1]
            V_grid = V_grid.T.reshape(2, ns, ns)

            # Mask low-confidence regions
            mass = np.sqrt((V_grid * V_grid).sum(0))
            min_mass = 1e-5
            min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
            cutoff1 = mass < min_mass

            length = np.sum(np.mean(np.abs(V[neighs]), axis=1), axis=1).reshape(ns, ns)
            cutoff2 = length < np.percentile(length, 5)

            cutoff = cutoff1 | cutoff2
            V_grid[0][cutoff] = np.nan
        else:
            # Format for quiver plot
            min_weight = np.percentile(weight_sum, 99) * 0.01
            E_grid = E_grid[weight_sum > min_weight]
            V_grid = V_grid[weight_sum > min_weight]
            V_grid /= 3 * quiver_autoscale(E_grid, V_grid)

        return E_grid, V_grid

    # ========================================================================
    # Utility Functions
    # ========================================================================

    def get_resource_metrics(self):
        """
        Get training resource usage metrics.
        
        Returns
        -------
        metrics : dict
            Dictionary with 'train_time', 'peak_memory_gb', 'actual_epochs'
        """
        return {
            'train_time': self.train_time,
            'peak_memory_gb': self.peak_memory_gb,
            'actual_epochs': self.actual_epochs
        }


# ============================================================================
# Helper Functions
# ============================================================================

def quiver_autoscale(E: np.ndarray, V: np.ndarray) -> float:
    """
    Compute autoscale factor for quiver/streamplot visualization.
    
    Parameters
    ----------
    E : np.ndarray
        Embedding coordinates, shape (n_cells, 2)
    V : np.ndarray
        Velocity vectors, shape (n_cells, 2)
    
    Returns
    -------
    scale : float
        Autoscale factor
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()
    
    # Avoid division by zero
    if scale_factor == 0:
        scale_factor = 1.0

    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles="xy",
        scale=None,
        scale_units="xy",
    )
    
    # Render the figure to compute Q.scale
    try:
        fig.canvas.draw()
        quiver_scale = Q.scale if Q.scale is not None else 1.0
    except Exception:
        # Fallback if rendering fails
        quiver_scale = 1.0
    finally:
        plt.close(fig)

    return quiver_scale / scale_factor


def l2_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute L2 norm (Euclidean length) of vectors.
    
    Parameters
    ----------
    x : np.ndarray or sparse matrix
        Input vectors
    axis : int, default=-1
        Axis along which to compute norm
    
    Returns
    -------
    norm : np.ndarray
        L2 norms
    """
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)  # type: ignore[attr-defined]
    else:
        return np.sqrt(np.sum(x * x, axis=axis))