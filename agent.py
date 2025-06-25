
from .environment import Env
from .utils import quiver_autoscale, l2_norm
import scanpy as sc
from anndata import AnnData
import torch
import tqdm
from typing import Optional, Literal, Tuple
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors


class agent(Env):
    """
    Agent class for single-cell RNA velocity analysis using iAODEVAE.
    
    This class extends the Env class to provide a complete framework for
    single-cell RNA velocity analysis, including model fitting, latent
    representation extraction, data imputation, and velocity field computation.
    
    The agent integrates variational autoencoders with ordinary differential
    equations to model cellular dynamics and compute RNA velocity fields
    for trajectory inference.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell expression data
    layer : str, optional
        Layer name in adata.layers to use for training data. Default is "counts".
    percent : float, optional
        Percentage of data to use in each batch (0.0 to 1.0). Default is 0.01.
    recon : float, optional
        Weight for reconstruction loss. Default is 1.0.
    irecon : float, optional
        Weight for information reconstruction loss. Default is 0.0.
    beta : float, optional
        Weight for KL divergence regularization. Default is 1.0.
    dip : float, optional
        Weight for disentangled information processing loss. Default is 0.0.
    tc : float, optional
        Weight for total correlation loss. Default is 0.0.
    info : float, optional
        Weight for information bottleneck loss (MMD). Default is 0.0.
    hidden_dim : int, optional
        Dimension of hidden layers in the network. Default is 128.
    latent_dim : int, optional
        Dimension of the latent representation space. Default is 10.
    i_dim : int, optional
        Dimension of the information bottleneck. Default is 2.
    use_ode : bool, optional
        Whether to enable ODE integration for temporal modeling. Default is False.
    loss_mode : Literal["mse", "nb", "zinb"], optional
        Type of loss function. Default is "nb".
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-4.
    vae_reg : float, optional
        Regularization weight for VAE component. Default is 0.5.
    ode_reg : float, optional
        Regularization weight for ODE component. Default is 0.5.
    device : torch.device, optional
        Device to run computations on. Default is CUDA if available, otherwise CPU.
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
        percent: float = 0.01,
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
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        # Initialize parent environment with all parameters
        super().__init__(
            adata=adata,
            layer=layer,
            percent=percent,
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
        )

    def fit(self, epochs: int = 1000) -> 'agent':
        """
        Train the iAODEVAE model for the specified number of epochs.
        
        This method performs iterative training with progress monitoring,
        displaying training loss and evaluation metrics including clustering
        performance and representation quality measures.
        
        Parameters
        ----------
        epochs : int, optional
            Number of training epochs. Default is 1000.
            
        Returns
        -------
        agent
            Returns self for method chaining
        """
        with tqdm.tqdm(total=int(epochs), desc="Fitting", ncols=150) as pbar:
            for i in range(int(epochs)):
                # Load batch data and perform training step
                data = self.load_data()
                self.step(data)
                
                # Update progress bar with metrics every 10 epochs
                if (i + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "Loss": f"{self.loss[-1][0]:.2f}",
                            "ARI": f"{(self.score[-1][0]):.2f}",
                            "NMI": f"{(self.score[-1][1]):.2f}",
                            "ASW": f"{(self.score[-1][2]):.2f}",
                            "C_H": f"{(self.score[-1][3]):.2f}",
                            "D_B": f"{(self.score[-1][4]):.2f}",
                            "P_C": f"{(self.score[-1][5]):.2f}",
                        }
                    )
                pbar.update(1)
        return self

    def get_iembed(self) -> np.ndarray:
        """
        Extract information bottleneck embeddings from all data.
        
        This method computes low-dimensional embeddings through the
        information bottleneck layer, which provides compressed
        representations suitable for visualization and downstream analysis.
        
        Returns
        -------
        np.ndarray
            Information bottleneck embeddings of shape (n_cells, i_dim)
        """
        iembed = self.take_iembed(self.X)
        return iembed

    def get_latent(self) -> np.ndarray:
        """
        Extract full latent representations from all data.
        
        This method computes the complete latent space representations
        learned by the variational autoencoder, capturing the full
        dimensionality of the learned manifold.
        
        Returns
        -------
        np.ndarray
            Latent representations of shape (n_cells, latent_dim)
        """
        latent = self.take_latent(self.X)
        return latent

    def get_time(self) -> np.ndarray:
        """
        Extract predicted pseudotime values from all data.
        
        This method computes pseudotime estimates when ODE mode is enabled,
        providing temporal ordering of cells along developmental trajectories.
        
        Returns
        -------
        np.ndarray
            Pseudotime values of shape (n_cells,)
            
        Notes
        -----
        This method requires use_ode=True during initialization.
        """
        time = self.take_time(self.X)
        return time

    def get_impute(
        self, 
        top_k: int = 30, 
        alpha: float = 0.9, 
        steps: int = 3, 
        decay: float = 0.99
    ) -> np.ndarray:
        """
        Perform data imputation using transition-based smoothing.
        
        This method computes imputed expression values by leveraging
        learned transition probabilities to smooth expression across
        similar cells in the latent space.
        
        Parameters
        ----------
        top_k : int, optional
            Number of top transitions to keep for sparsification. Default is 30.
        alpha : float, optional
            Blending weight between original and imputed data. Default is 0.9.
        steps : int, optional
            Number of diffusion steps for multi-step imputation. Default is 3.
        decay : float, optional
            Decay factor for multi-step contributions. Default is 0.99.
            
        Returns
        -------
        np.ndarray
            Imputed expression data of shape (n_cells, n_genes)
        """
        # Compute transition probability matrix
        T = self.take_transition(self.X, top_k)

        def multi_step_impute(T: np.ndarray, X: np.ndarray, steps: int, decay: float) -> np.ndarray:
            """
            Perform multi-step imputation using iterative diffusion.
            
            Parameters
            ----------
            T : np.ndarray
                Transition probability matrix
            X : np.ndarray
                Original expression data
            steps : int
                Number of diffusion steps
            decay : float
                Decay factor for step contributions
                
            Returns
            -------
            np.ndarray
                Multi-step imputed data
            """
            X_current = X.copy()
            X_imputed = X.copy()
            
            # Iteratively apply transition matrix with decay
            for i in range(steps):
                X_current = T @ X_current
                X_imputed = X_imputed + decay**i * X_current
                
            # Normalize by total weight
            X_imputed = X_imputed / (1 + sum(decay**i for i in range(steps)))
            return X_imputed

        def balanced_impute(
            T: np.ndarray, 
            X: np.ndarray, 
            alpha: float = 0.5, 
            steps: int = 3, 
            decay: float = 0.9
        ) -> np.ndarray:
            """
            Create balanced imputation combining original and smoothed data.
            
            Parameters
            ----------
            T : np.ndarray
                Transition probability matrix
            X : np.ndarray
                Original expression data
            alpha : float
                Blending weight (0=original, 1=fully imputed)
            steps : int
                Number of diffusion steps
            decay : float
                Decay factor for step contributions
                
            Returns
            -------
            np.ndarray
                Balanced imputed data
            """
            X_imputed = multi_step_impute(T, X, steps, decay)
            X_balanced = (1 - alpha) * X + alpha * X_imputed
            return X_balanced

        return balanced_impute(T, self.X, alpha, steps, decay)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute complete velocity field results including grid representation.
        
        This method performs the full velocity field computation pipeline:
        1. Computes gradients from the ODE solver
        2. Calculates similarity matrix based on gradient-embedding alignment
        3. Derives velocity field from transition probabilities
        4. Creates grid-based representation for visualization
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object to store results
        zs_key : str
            Key for latent space embeddings in adata.obsm
        E_key : str
            Key for 2D embeddings (e.g., UMAP) in adata.obsm
        vf_key : str, optional
            Key to store velocity field in adata.obsm. Default is "X_vf".
        T_key : str, optional
            Key to store transition matrix in adata.obsp. Default is "cosine_similarity".
        dv_key : str, optional
            Key to store derived velocity in adata.obsm. Default is "X_dv".
        reverse : bool, optional
            Whether to reverse velocity direction. Default is False.
        run_neigh : bool, optional
            Whether to recompute neighborhood graph. Default is True.
        use_rep_neigh : Optional[str], optional
            Representation to use for neighborhood computation. Default is None.
        t_key : Optional[str], optional
            Key for pseudotime in adata.obs. Default is None.
        n_neigh : int, optional
            Number of neighbors for graph construction. Default is 20.
        var_stabilize_transform : bool, optional
            Whether to apply variance stabilizing transform. Default is False.
        scale : int, optional
            Scaling factor for transition probabilities. Default is 10.
        self_transition : bool, optional
            Whether to include self-transitions. Default is False.
        smooth : float, optional
            Smoothing parameter for grid interpolation. Default is 0.5.
        stream : bool, optional
            Whether to create streamplot-compatible output. Default is True.
        density : float, optional
            Grid density factor. Default is 1.0.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - E_grid: Grid coordinates for visualization
            - V_grid: Velocity field on the grid
        """
        # Compute gradients from ODE solver
        grads = self.take_grad(self.X)
        adata.obsm[vf_key] = grads
        
        # Compute similarity matrix
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
        
        # Derive velocity field
        adata.obsm[dv_key] = self.get_vf(
            adata,
            T_key=T_key,
            E_key=E_key,
            scale=scale,
            self_transition=self_transition,
        )
        
        # Create grid representation
        E = np.array(adata.obsm[E_key])
        V = adata.obsm[dv_key]
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
    ) -> csr_matrix:
        """
        Compute cosine similarity matrix between cell embeddings and velocity directions.
        
        This method calculates the alignment between local embedding differences
        and predicted velocity directions to construct a transition probability matrix.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        zs_key : str
            Key for latent space embeddings in adata.obsm
        vf_key : str, optional
            Key for velocity field in adata.obsm. Default is "X_vf".
        reverse : bool, optional
            Whether to reverse velocity direction. Default is False.
        run_neigh : bool, optional
            Whether to recompute neighborhood graph. Default is True.
        use_rep_neigh : Optional[str], optional
            Representation to use for neighborhood computation. Default is None.
        t_key : Optional[str], optional
            Key for pseudotime-based neighbor selection. Default is None.
        n_neigh : int, optional
            Number of neighbors for graph construction. Default is 20.
        var_stabilize_transform : bool, optional
            Whether to apply variance stabilizing transform. Default is False.
            
        Returns
        -------
        csr_matrix
            Cosine similarity matrix of shape (n_cells, n_cells)
        """
        Z = np.array(adata.obsm[zs_key])
        V = np.array(adata.obsm[vf_key])
        
        # Apply optional transformations
        if reverse:
            V = -V
        if var_stabilize_transform:
            V = np.sqrt(np.abs(V)) * np.sign(V)

        ncells = adata.n_obs

        # Setup neighborhood graph
        if run_neigh or ("neighbors" not in adata.uns):
            if use_rep_neigh is None:
                use_rep_neigh = zs_key
            else:
                if use_rep_neigh not in adata.obsm:
                    raise KeyError(
                        f"`{use_rep_neigh}` not found in `.obsm` of the AnnData. "
                        f"Please provide valid `use_rep_neigh` for neighbor detection."
                    )
            sc.pp.neighbors(adata, use_rep=use_rep_neigh, n_neighbors=n_neigh)
            
        n_neigh = adata.uns["neighbors"]["params"]["n_neighbors"] - 1

        # Setup pseudotime-based neighbors if specified
        if t_key is not None:
            if t_key not in adata.obs:
                raise KeyError(
                    f"`{t_key}` not found in `.obs` of the AnnData. "
                    f"Please provide valid `t_key` for estimated pseudotime."
                )
            ts = adata.obs[t_key].values
            indices_matrix2 = np.zeros((ncells, n_neigh), dtype=int)
            
            for i in range(ncells):
                idx = np.abs(ts - ts[i]).argsort()[: (n_neigh + 1)]
                idx = np.setdiff1d(idx, i) if i in idx else idx[:-1]
                indices_matrix2[i] = idx

        # Compute cosine similarities
        vals, rows, cols = [], [], []
        
        for i in range(ncells):
            # Get primary neighbors
            idx = adata.obsp["distances"][i].indices
            
            # Get secondary neighbors (neighbors of neighbors)
            idx2 = adata.obsp["distances"][idx].indices
            idx2 = np.setdiff1d(idx2, i)
            
            # Combine neighbor sets
            idx = (
                np.unique(np.concatenate([idx, idx2]))
                if t_key is None
                else np.unique(np.concatenate([idx, idx2, indices_matrix2[i]]))
            )
            
            # Compute embedding differences
            dZ = Z[idx] - Z[i, None]
            if var_stabilize_transform:
                dZ = np.sqrt(np.abs(dZ)) * np.sign(dZ)
                
            # Compute cosine similarity with velocity
            cos_sim = np.einsum("ij, j", dZ, V[i]) / (
                l2_norm(dZ, axis=1) * l2_norm(V[i])
            )
            cos_sim[np.isnan(cos_sim)] = 0
            
            # Store results
            vals.extend(cos_sim)
            rows.extend(np.repeat(i, len(idx)))
            cols.extend(idx)

        # Create sparse matrix
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
    ) -> np.ndarray:
        """
        Derive velocity field from transition probabilities and embeddings.
        
        This method computes cell-specific velocity vectors by analyzing
        the expected movement in embedding space based on transition probabilities.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        T_key : str
            Key for transition matrix in adata.obsp
        E_key : str
            Key for 2D embeddings in adata.obsm
        scale : int, optional
            Scaling factor for transition probabilities. Default is 10.
        self_transition : bool, optional
            Whether to include self-transitions. Default is False.
            
        Returns
        -------
        np.ndarray
            Velocity field vectors of shape (n_cells, n_embedding_dims)
        """
        T = adata.obsp[T_key].copy()

        # Handle self-transitions
        if self_transition:
            max_t = T.max(1).A.flatten()
            ub = np.percentile(max_t, 98)
            self_t = np.clip(ub - max_t, 0, 1)
            T.setdiag(self_t)

        # Apply exponential scaling and normalization
        T = T.sign().multiply(np.expm1(abs(T * scale)))
        T = T.multiply(csr_matrix(1.0 / abs(T).sum(1)))
        
        if self_transition:
            T.setdiag(0)
            T.eliminate_zeros()

        # Extract embeddings and initialize velocity
        E = np.array(adata.obsm[E_key])
        V = np.zeros(E.shape)

        # Compute velocity for each cell
        for i in range(adata.n_obs):
            idx = T[i].indices
            
            # Compute embedding differences
            dE = E[idx] - E[i, None]
            dE /= l2_norm(dE)[:, None]
            dE[np.isnan(dE)] = 0
            
            # Weight by transition probabilities
            prob = T[i].data
            V[i] = prob.dot(dE) - prob.mean() * dE.sum(0)

        # Normalize velocity magnitude
        V /= 3 * quiver_autoscale(E, V)
        return V

    def get_vfgrid(
        self,
        E: np.ndarray,
        V: np.ndarray,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create grid-based representation of velocity field for visualization.
        
        This method interpolates the discrete velocity field onto a regular
        grid suitable for streamplot visualization or quiver plots.
        
        Parameters
        ----------
        E : np.ndarray
            Cell embeddings of shape (n_cells, n_dims)
        V : np.ndarray
            Velocity vectors of shape (n_cells, n_dims)
        smooth : float, optional
            Smoothing parameter for interpolation. Default is 0.5.
        stream : bool, optional
            Whether to format output for streamplot. Default is True.
        density : float, optional
            Grid density factor. Default is 1.0.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - E_grid: Grid coordinates
            - V_grid: Velocity field on the grid
        """
        # Create coordinate grids for each dimension
        grs = []
        for i in range(E.shape[1]):
            m, M = np.min(E[:, i]), np.max(E[:, i])
            diff = M - m
            m = m - 0.01 * diff
            M = M + 0.01 * diff
            gr = np.linspace(m, M, int(50 * density))
            grs.append(gr)

        # Create meshgrid and flatten
        meshes = np.meshgrid(*grs)
        E_grid = np.vstack([i.flat for i in meshes]).T

        # Setup nearest neighbor interpolation
        n_neigh = int(E.shape[0] / 50)
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1)
        nn.fit(E)
        dists, neighs = nn.kneighbors(E_grid)

        # Compute interpolation weights
        scale = np.mean([g[1] - g[0] for g in grs]) * smooth
        weight = norm.pdf(x=dists, scale=scale)
        weight_sum = weight.sum(1)

        # Interpolate velocity field
        V_grid = (V[neighs] * weight[:, :, None]).sum(1)
        V_grid /= np.maximum(1, weight_sum)[:, None]

        if stream:
            # Format for streamplot
            E_grid = np.stack(grs)
            ns = E_grid.shape[1]
            V_grid = V_grid.T.reshape(2, ns, ns)

            # Apply quality filters
            mass = np.sqrt((V_grid * V_grid).sum(0))
            min_mass = 1e-5
            min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
            cutoff1 = mass < min_mass

            length = np.sum(np.mean(np.abs(V[neighs]), axis=1), axis=1).reshape(ns, ns)
            cutoff2 = length < np.percentile(length, 5)

            # Mask low-quality regions
            cutoff = cutoff1 | cutoff2
            V_grid[0][cutoff] = np.nan
        else:
            # Filter by interpolation quality
            min_weight = np.percentile(weight_sum, 99) * 0.01
            E_grid, V_grid = (
                E_grid[weight_sum > min_weight],
                V_grid[weight_sum > min_weight],
            )
            V_grid /= 3 * quiver_autoscale(E_grid, V_grid)

        return E_grid, V_grid

