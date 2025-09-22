
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from typing import Tuple, Optional, Literal
from warnings import warn

from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE


class iAODEVAE(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    Interpretable ATAC-seq Ordinary Differential Equation Variational Autoencoder.
    
    A specialized variational autoencoder designed for single-cell ATAC-seq (scATAC-seq) data
    analysis that integrates multiple advanced techniques:
    
    - **i** (Interpretable): Information bottleneck for interpretable latent representations
    - **A** (ATAC-seq): Optimized for single-cell ATAC-seq chromatin accessibility data  
    - **ODE** (Ordinary Differential Equations): Neural ODE integration for temporal dynamics
    - **VAE** (Variational Autoencoder): Probabilistic latent variable modeling
    
    This model combines several state-of-the-art regularization techniques:
    - Disentangled Information Processing (DIP) for factor separation
    - Beta-Total Correlation (β-TC) for disentangled representations  
    - Information bottleneck constraints for interpretability
    - Optional Neural ODE integration for continuous-time cellular dynamics
    - Multiple probabilistic distributions suitable for scATAC-seq count data
    
    Parameters
    ----------
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
        Weight coefficient for information bottleneck loss (MMD regularization)
    state_dim : int
        Dimension of scATAC-seq input space (number of accessible chromatin peaks)
    hidden_dim : int
        Dimension of hidden layers in encoder/decoder networks
    latent_dim : int
        Dimension of the primary latent representation space
    i_dim : int
        Dimension of the interpretable information bottleneck layer
    use_ode : bool
        Whether to enable Neural ODE integration for temporal cellular dynamics
    loss_mode : Literal["mse", "nb", "zinb"]
        Probabilistic distribution for scATAC-seq data:
        - "mse": Gaussian (continuous approximation)
        - "nb": Negative Binomial (count data) 
        - "zinb": Zero-Inflated Negative Binomial (sparse count data, recommended for scATAC-seq)
    lr : float
        Learning rate for Adam optimizer
    vae_reg : float
        Regularization weight for standard VAE latent representations
    ode_reg : float  
        Regularization weight for ODE-integrated latent representations
    device : torch.device
        Computing device (CPU/CUDA)
    
    Attributes
    ----------
    nn : VAE
        The underlying VAE neural network architecture
    nn_optimizer : torch.optim.Adam
        Adam optimizer for network parameters
    loss : list
        Training loss history tracking
    
    Notes
    -----
    For scATAC-seq data, "zinb" loss mode is typically recommended due to the sparse, 
    zero-inflated nature of chromatin accessibility measurements. The ODE integration
    can capture cellular trajectory dynamics in the latent space.
    
    Examples
    --------
    >>> # Initialize for scATAC-seq with 50k peaks, ODE dynamics, ZINB loss
    >>> model = iAODEVAE(
    ...     recon=1.0, irecon=0.5, beta=1.0, dip=0.1, tc=0.1, info=0.1,
    ...     state_dim=50000, hidden_dim=512, latent_dim=64, i_dim=16,
    ...     use_ode=True, loss_mode="zinb", lr=1e-3, 
    ...     vae_reg=0.7, ode_reg=0.3, device=torch.device("cuda")
    ... )
    """
    
    def __init__(
        self,
        recon: float,
        irecon: float,
        beta: float,
        dip: float,
        tc: float,
        info: float,
        state_dim: int,
        hidden_dim: int,
        latent_dim: int,
        i_dim: int,
        use_ode: bool,
        loss_mode: Literal["mse", "nb", "zinb"],
        lr: float,
        vae_reg: float,
        ode_reg: float,
        device: torch.device,
        *args,
        **kwargs,
    ) -> None:
        
        # Validate inputs
        self._validate_hyperparameters(
            recon, irecon, beta, dip, tc, info, lr, vae_reg, ode_reg
        )
        self._validate_architecture_params(
            state_dim, hidden_dim, latent_dim, i_dim
        )
        
        # Store model configuration
        self.use_ode = use_ode
        self.loss_mode = loss_mode
        self.device = device
        
        # Loss weights
        self.recon = recon
        self.irecon = irecon
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        
        # ODE regularization weights
        if use_ode:
            if abs(vae_reg + ode_reg - 1.0) > 1e-6:
                warn(f"VAE and ODE regularization weights should sum to 1.0, got {vae_reg + ode_reg}")
        self.vae_reg = vae_reg
        self.ode_reg = ode_reg
        
        # Initialize VAE architecture
        self.nn = VAE(
            state_dim=state_dim,
            hidden_dim=hidden_dim, 
            action_dim=latent_dim,  # Using action_dim for latent_dim compatibility
            i_dim=i_dim,
            use_ode=use_ode,
            loss_mode=loss_mode,
            device=device
        )
        
        # Initialize optimizer
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        
        # Loss tracking
        self.loss = []
        
    def _validate_hyperparameters(
        self, recon: float, irecon: float, beta: float, dip: float, 
        tc: float, info: float, lr: float, vae_reg: float, ode_reg: float
    ) -> None:
        """Validate hyperparameter values."""
        if any(param < 0 for param in [recon, irecon, beta, dip, tc, info]):
            raise ValueError("All loss weights must be non-negative")
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if any(reg < 0 or reg > 1 for reg in [vae_reg, ode_reg]):
            raise ValueError("Regularization weights must be in [0, 1]")
            
    def _validate_architecture_params(
        self, state_dim: int, hidden_dim: int, latent_dim: int, i_dim: int
    ) -> None:
        """Validate architecture dimensions."""
        dims = {"state_dim": state_dim, "hidden_dim": hidden_dim, 
                "latent_dim": latent_dim, "i_dim": i_dim}
        for name, dim in dims.items():
            if dim <= 0:
                raise ValueError(f"{name} must be positive, got {dim}")
        if i_dim > latent_dim:
            warn(f"Information bottleneck dimension ({i_dim}) larger than latent dimension ({latent_dim})")

    @torch.no_grad()
    def extract_latent_representations(self, atac_data: np.ndarray) -> np.ndarray:
        """
        Extract latent representations from scATAC-seq data.
        
        For ODE mode, combines standard VAE latents with ODE-integrated dynamics.
        For standard mode, returns encoded latent representations.
        
        Parameters
        ----------
        atac_data : np.ndarray
            scATAC-seq count matrix of shape (n_cells, n_peaks)
            
        Returns
        -------
        np.ndarray
            Latent representations of shape (n_cells, latent_dim)
            
        Notes
        -----
        In ODE mode, latent representations are weighted combinations:
        latent = vae_reg * z_vae + ode_reg * z_ode
        """
        atac_tensor = torch.tensor(atac_data, dtype=torch.float32).to(self.device)
        
        if self.use_ode:
            # Extract temporal dynamics through ODE integration
            q_z, q_m, q_s, t = self.nn.encoder(atac_tensor)
            t_cpu = t.cpu().numpy()
            
            # Sort by pseudotime and integrate ODE
            t_sorted, sort_indices, inverse_indices = np.unique(
                t_cpu, return_index=True, return_inverse=True
            )
            t_tensor = torch.tensor(t_sorted, dtype=torch.float32).to(self.device)
            q_z_sorted = q_z[sort_indices]
            
            # Integrate from initial condition
            z_initial = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z_initial, t_tensor)
            q_z_ode_reordered = q_z_ode[inverse_indices]
            
            # Combine VAE and ODE latents
            combined_latent = (self.vae_reg * q_z + self.ode_reg * q_z_ode_reordered)
            return combined_latent.cpu().numpy()
        else:
            # Standard VAE encoding
            q_z, q_m, q_s = self.nn.encoder(atac_tensor)
            return q_z.cpu().numpy()

    @torch.no_grad() 
    def extract_interpretable_embeddings(self, atac_data: np.ndarray) -> np.ndarray:
        """
        Extract interpretable embeddings through information bottleneck layer.
        
        The information bottleneck creates compressed, interpretable representations
        by forcing information through a narrower dimensional space (i_dim < latent_dim).
        
        Parameters
        ----------
        atac_data : np.ndarray
            scATAC-seq count matrix of shape (n_cells, n_peaks)
            
        Returns
        -------
        np.ndarray
            Interpretable embeddings of shape (n_cells, i_dim)
            
        Notes
        -----
        These embeddings are designed to capture the most essential regulatory
        features while discarding noise, making them suitable for downstream
        interpretation and visualization of chromatin accessibility patterns.
        """
        atac_tensor = torch.tensor(atac_data, dtype=torch.float32).to(self.device)
        
        if self.use_ode:
            # ODE mode: combine temporal dynamics with information bottleneck
            q_z, q_m, q_s, t = self.nn.encoder(atac_tensor)
            t_cpu = t.cpu().numpy()
            
            # Process temporal sorting and ODE integration
            t_sorted, sort_indices, inverse_indices = np.unique(
                t_cpu, return_index=True, return_inverse=True
            )
            t_tensor = torch.tensor(t_sorted, dtype=torch.float32).to(self.device)
            q_z_sorted = q_z[sort_indices]
            
            # Integrate ODE dynamics
            z_initial = q_z_sorted[0]  
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z_initial, t_tensor)
            q_z_ode_reordered = q_z_ode[inverse_indices]

            # Apply information bottleneck to both representations
            embed_vae = self.nn.latent_encoder(q_z)
            embed_ode = self.nn.latent_encoder(q_z_ode_reordered)
            
            # Weighted combination of interpretable embeddings
            combined_embed = (self.vae_reg * embed_vae + self.ode_reg * embed_ode)
            return combined_embed.cpu().numpy()
        else:
            # Standard mode: direct information bottleneck encoding
            forward_output = self.nn(atac_tensor)
            if self.loss_mode == "zinb":
                # ZINB mode: (q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl)
                interpretable_embed = forward_output[5]  # le (latent encoded)
            else:
                # MSE/NB mode: (q_z, q_m, q_s, pred_x, le, pred_xl) 
                interpretable_embed = forward_output[4]  # le (latent encoded)
            
            return interpretable_embed.cpu().numpy()

    @torch.no_grad()
    def extract_pseudotime(self, atac_data: np.ndarray) -> np.ndarray:
        """
        Extract predicted pseudotime parameters from scATAC-seq data.
        
        Only available in ODE mode. Pseudotime represents the continuous temporal
        progression of cellular states in chromatin accessibility space.
        
        Parameters
        ----------
        atac_data : np.ndarray
            scATAC-seq count matrix of shape (n_cells, n_peaks)
            
        Returns
        -------
        np.ndarray  
            Predicted pseudotime values of shape (n_cells,) in range [0, 1]
            
        Raises
        ------
        ValueError
            If called when use_ode=False
            
        Notes
        -----
        Pseudotime values are normalized to [0, 1] range where 0 represents
        the earliest cellular state and 1 represents the most differentiated state
        in the chromatin accessibility trajectory.
        """
        if not self.use_ode:
            raise ValueError("Pseudotime extraction requires ODE mode (use_ode=True)")
            
        atac_tensor = torch.tensor(atac_data, dtype=torch.float32).to(self.device)
        _, _, _, pseudotime = self.nn.encoder(atac_tensor)
        return pseudotime.detach().cpu().numpy()

    @torch.no_grad()
    def extract_velocity_gradients(self, atac_data: np.ndarray) -> np.ndarray:
        """
        Compute velocity gradients from the Neural ODE for trajectory analysis.
        
        The gradients represent the rate of change in latent chromatin accessibility
        patterns, analogous to RNA velocity but for ATAC-seq data.
        
        Parameters
        ----------
        atac_data : np.ndarray
            scATAC-seq count matrix of shape (n_cells, n_peaks)
            
        Returns
        -------
        np.ndarray
            Velocity gradients of shape (n_cells, latent_dim)
            
        Raises
        ------
        ValueError
            If called when use_ode=False
            
        Notes
        -----
        These gradients indicate the direction and magnitude of chromatin
        accessibility changes, enabling velocity field visualization and
        trajectory inference in the latent space.
        """
        if not self.use_ode:
            raise ValueError("Velocity gradients require ODE mode (use_ode=True)")
            
        atac_tensor = torch.tensor(atac_data, dtype=torch.float32).to(self.device)
        q_z, q_m, q_s, pseudotime = self.nn.encoder(atac_tensor)
        
        # Compute gradients using ODE solver
        velocity_gradients = self.nn.ode_solver(pseudotime, q_z)
        return velocity_gradients.detach().cpu().numpy()

    @torch.no_grad()
    def compute_transition_probabilities(
        self, atac_data: np.ndarray, top_k: int = 30, time_step: float = 1e-2
    ) -> np.ndarray:
        """
        Compute transition probability matrix for cellular trajectory analysis.
        
        Uses Neural ODE dynamics to predict future cellular states and compute
        similarity-based transition probabilities between cells.
        
        Parameters
        ----------
        atac_data : np.ndarray
            scATAC-seq count matrix of shape (n_cells, n_peaks)
        top_k : int, optional
            Number of top transitions to retain per cell for sparsification. Default is 30.
        time_step : float, optional
            Integration time step for future state prediction. Default is 1e-2.
            
        Returns
        -------
        np.ndarray
            Sparse transition probability matrix of shape (n_cells, n_cells)
            
        Raises
        ------
        ValueError
            If called when use_ode=False or invalid parameters
            
        Notes
        -----
        The transition matrix T[i,j] represents the probability of cell i 
        transitioning to cell j based on predicted chromatin accessibility dynamics.
        Larger time_step values predict further into the future but may be less accurate.
        """
        if not self.use_ode:
            raise ValueError("Transition probabilities require ODE mode (use_ode=True)")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if time_step <= 0:
            raise ValueError(f"time_step must be positive, got {time_step}")
            
        atac_tensor = torch.tensor(atac_data, dtype=torch.float32).to(self.device)
        q_z, q_m, q_s, pseudotime = self.nn.encoder(atac_tensor)
        
        # Compute velocity gradients and predict future states
        velocity_grads = self.nn.ode_solver(pseudotime, q_z).detach().cpu().numpy()
        current_states = q_z.detach().cpu().numpy()
        future_states = current_states + time_step * velocity_grads
        
        # Compute pairwise distances and similarities
        distances = pairwise_distances(current_states, future_states, metric='euclidean')
        
        # Adaptive bandwidth selection
        median_distance = np.median(distances[distances > 0])
        bandwidth = median_distance if median_distance > 0 else 1.0
        
        # Gaussian kernel similarities
        similarities = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))
        
        # Normalize to probability distributions
        transition_matrix = similarities / (similarities.sum(axis=1, keepdims=True) + 1e-12)
        
        # Sparsify to top-k transitions per cell
        return self._sparsify_transition_matrix(transition_matrix, top_k)

    def _sparsify_transition_matrix(
        self, transition_matrix: np.ndarray, top_k: int
    ) -> np.ndarray:
        """
        Sparsify transition matrix by retaining only top-k transitions per cell.
        
        Parameters
        ---------- 
        transition_matrix : np.ndarray
            Full transition probability matrix
        top_k : int
            Number of top transitions to retain per cell
            
        Returns
        -------
        np.ndarray
            Sparsified transition matrix
        """
        n_cells = transition_matrix.shape[0]
        sparse_matrix = np.zeros_like(transition_matrix)
        
        for i in range(n_cells):
            # Find top-k highest probability transitions
            top_indices = np.argpartition(transition_matrix[i], -top_k)[-top_k:]
            sparse_matrix[i, top_indices] = transition_matrix[i, top_indices]
            
            # Renormalize to maintain probability distribution
            row_sum = sparse_matrix[i].sum()
            if row_sum > 0:
                sparse_matrix[i] /= row_sum
                
        return sparse_matrix

    def train_step(self, atac_batch: np.ndarray) -> dict:
        """
        Perform one training iteration on a batch of scATAC-seq data.
        
        Parameters
        ----------
        atac_batch : np.ndarray
            Batch of scATAC-seq count data of shape (batch_size, n_peaks)
            
        Returns
        -------
        dict
            Dictionary containing individual loss components for monitoring:
            - 'total_loss': Combined loss value
            - 'reconstruction': Primary reconstruction loss  
            - 'info_reconstruction': Information bottleneck reconstruction loss
            - 'kl_divergence': KL regularization loss
            - 'dip_loss': Disentanglement loss
            - 'tc_loss': Total correlation loss
            - 'mmd_loss': Maximum mean discrepancy loss
            - 'ode_consistency': ODE consistency loss (if use_ode=True)
        """
        atac_tensor = torch.tensor(atac_batch, dtype=torch.float32).to(self.device)
        
        # Forward pass and loss computation
        loss_dict = self._compute_total_loss(atac_tensor)
        total_loss = loss_dict['total_loss']
        
        # Optimization step
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), max_norm=1.0)  # Gradient clipping
        self.nn_optimizer.step()
        
        # Convert to CPU values for logging
        loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in loss_dict.items()}
        
        # Store loss history
        loss_tuple = (
            loss_values['total_loss'],
            loss_values['reconstruction'], 
            loss_values['info_reconstruction'],
            loss_values['kl_divergence'],
            loss_values['dip_loss'],
            loss_values['tc_loss'], 
            loss_values['mmd_loss']
        )
        if 'ode_consistency' in loss_values:
            loss_tuple += (loss_values['ode_consistency'],)
            
        self.loss.append(loss_tuple)
        
        return loss_values

    def _compute_total_loss(self, atac_data: torch.Tensor) -> dict:
        """Compute all loss components for the current batch."""
        loss_dict = {}
        
        if self.use_ode:
            loss_dict = self._compute_ode_losses(atac_data)
        else:
            loss_dict = self._compute_standard_losses(atac_data)
            
        return loss_dict

    def _compute_ode_losses(self, atac_data: torch.Tensor) -> dict:
        """Compute losses for ODE-enabled mode."""
        forward_output = self.nn(atac_data)
        
        if self.loss_mode == "zinb":
            (q_z, q_m, q_s, x_reordered, pred_x, dropout_logits, le, le_ode,
             pred_xl, dropout_logitsl, q_z_ode, pred_x_ode, 
             dropout_logits_ode, pred_xl_ode, dropout_logitsl_ode) = forward_output
        else:
            (q_z, q_m, q_s, x_reordered, pred_x, le, le_ode, pred_xl,
             q_z_ode, pred_x_ode, pred_xl_ode) = forward_output

        # ODE consistency loss
        ode_consistency = F.mse_loss(q_z, q_z_ode, reduction='none').sum(-1).mean()
        
        # Reconstruction losses
        recon_loss = self._compute_reconstruction_loss(
            x_reordered, pred_x, pred_x_ode, dropout_logits if self.loss_mode == "zinb" else None,
            dropout_logits_ode if self.loss_mode == "zinb" else None
        )
        
        irecon_loss = self._compute_info_reconstruction_loss(
            x_reordered, pred_xl, pred_xl_ode, 
            dropout_logitsl if self.loss_mode == "zinb" else None,
            dropout_logitsl_ode if self.loss_mode == "zinb" else None
        )
        
        # Regularization losses
        kl_div = self._compute_kl_loss(q_m, q_s)
        dip_loss = self._compute_dip_loss(q_m, q_s)
        tc_loss = self._compute_tc_loss(q_z, q_m, q_s)
        mmd_loss = self._compute_mmd_loss(q_z)
        
        # Total loss
        total_loss = (
            self.recon * recon_loss + irecon_loss + ode_consistency +
            kl_div + dip_loss + tc_loss + mmd_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction': recon_loss,
            'info_reconstruction': irecon_loss,
            'ode_consistency': ode_consistency,
            'kl_divergence': kl_div,
            'dip_loss': dip_loss,
            'tc_loss': tc_loss,
            'mmd_loss': mmd_loss
        }

    def _compute_standard_losses(self, atac_data: torch.Tensor) -> dict:
        """Compute losses for standard VAE mode."""
        forward_output = self.nn(atac_data)
        
        if self.loss_mode == "zinb":
            (q_z, q_m, q_s, pred_x, dropout_logits, 
             le, pred_xl, dropout_logitsl) = forward_output
        else:
            (q_z, q_m, q_s, pred_x, le, pred_xl) = forward_output

        # Reconstruction losses  
        recon_loss = self._compute_reconstruction_loss(
            atac_data, pred_x, None, 
            dropout_logits if self.loss_mode == "zinb" else None, None
        )
        
        irecon_loss = self._compute_info_reconstruction_loss(
            atac_data, pred_xl, None,
            dropout_logitsl if self.loss_mode == "zinb" else None, None
        )
        
        # Regularization losses
        kl_div = self._compute_kl_loss(q_m, q_s)
        dip_loss = self._compute_dip_loss(q_m, q_s)
        tc_loss = self._compute_tc_loss(q_z, q_m, q_s)
        mmd_loss = self._compute_mmd_loss(q_z)
        
        # Total loss
        total_loss = (
            self.recon * recon_loss + irecon_loss +
            kl_div + dip_loss + tc_loss + mmd_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction': recon_loss, 
            'info_reconstruction': irecon_loss,
            'kl_divergence': kl_div,
            'dip_loss': dip_loss,
            'tc_loss': tc_loss,
            'mmd_loss': mmd_loss
        }

    def _compute_reconstruction_loss(
        self, targets: torch.Tensor, pred_x: torch.Tensor, 
        pred_x_ode: Optional[torch.Tensor] = None,
        dropout_logits: Optional[torch.Tensor] = None,
        dropout_logits_ode: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute reconstruction loss based on the specified mode."""
        if self.loss_mode == "zinb":
            return self._compute_zinb_loss(
                targets, pred_x, pred_x_ode, dropout_logits, dropout_logits_ode
            )
        elif self.loss_mode == "nb":
            return self._compute_nb_loss(targets, pred_x, pred_x_ode)
        else:  # mse
            return self._compute_mse_loss(targets, pred_x, pred_x_ode)

    def _compute_info_reconstruction_loss(
        self, targets: torch.Tensor, pred_xl: torch.Tensor,
        pred_xl_ode: Optional[torch.Tensor] = None,
        dropout_logitsl: Optional[torch.Tensor] = None,
        dropout_logitsl_ode: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute information bottleneck reconstruction loss."""
        if self.irecon == 0:
            return torch.tensor(0.0).to(self.device)
            
        if self.loss_mode == "zinb":
            return self.irecon * self._compute_zinb_loss(
                targets, pred_xl, pred_xl_ode, dropout_logitsl, dropout_logitsl_ode
            )
        elif self.loss_mode == "nb":
            return self.irecon * self._compute_nb_loss(targets, pred_xl, pred_xl_ode)
        else:  # mse
            return self.irecon * self._compute_mse_loss(targets, pred_xl, pred_xl_ode)

    def _compute_zinb_loss(
        self, targets: torch.Tensor, pred_x: torch.Tensor,
        pred_x_ode: Optional[torch.Tensor], dropout_logits: torch.Tensor,
        dropout_logits_ode: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute Zero-Inflated Negative Binomial loss."""
        # Scale predictions by library size
        library_size = targets.sum(-1).view(-1, 1)
        pred_x_scaled = pred_x * library_size
        
        # Dispersion parameter
        dispersion = torch.exp(self.nn.decoder.disp)
        
        # Primary ZINB loss
        zinb_loss = -self._log_zinb(targets, pred_x_scaled, dispersion, dropout_logits).sum(-1).mean()
        
        # Add ODE component if available
        if pred_x_ode is not None and dropout_logits_ode is not None:
            pred_x_ode_scaled = pred_x_ode * library_size
            zinb_loss += -self._log_zinb(
                targets, pred_x_ode_scaled, dispersion, dropout_logits_ode
            ).sum(-1).mean()
            
        return zinb_loss

    def _compute_nb_loss(
        self, targets: torch.Tensor, pred_x: torch.Tensor,
        pred_x_ode: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute Negative Binomial loss."""
        # Scale predictions by library size  
        library_size = targets.sum(-1).view(-1, 1)
        pred_x_scaled = pred_x * library_size
        
        # Dispersion parameter
        dispersion = torch.exp(self.nn.decoder.disp)
        
        # Primary NB loss
        nb_loss = -self._log_nb(targets, pred_x_scaled, dispersion).sum(-1).mean()
        
        # Add ODE component if available
        if pred_x_ode is not None:
            pred_x_ode_scaled = pred_x_ode * library_size
            nb_loss += -self._log_nb(targets, pred_x_ode_scaled, dispersion).sum(-1).mean()
            
        return nb_loss

    def _compute_mse_loss(
        self, targets: torch.Tensor, pred_x: torch.Tensor,
        pred_x_ode: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute Mean Squared Error loss."""
        mse_loss = F.mse_loss(targets, pred_x, reduction='none').sum(-1).mean()
        
        if pred_x_ode is not None:
            mse_loss += F.mse_loss(targets, pred_x_ode, reduction='none').sum(-1).mean()
            
        return mse_loss

    def _compute_kl_loss(self, q_mean: torch.Tensor, q_log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        if self.beta == 0:
            return torch.tensor(0.0).to(self.device)
        
        prior_mean = torch.zeros_like(q_mean)
        prior_log_var = torch.zeros_like(q_log_var)
        kl_loss = self.beta * self._normal_kl(q_mean, q_log_var, prior_mean, prior_log_var).sum(-1).mean()
        return kl_loss

    def _compute_dip_loss(self, q_mean: torch.Tensor, q_log_var: torch.Tensor) -> torch.Tensor:
        """Compute Disentangled Information Processing loss."""
        if self.dip == 0:
            return torch.tensor(0.0).to(self.device)
        return self.dip * self._dip_loss(q_mean, q_log_var)

    def _compute_tc_loss(
        self, q_z: torch.Tensor, q_mean: torch.Tensor, q_log_var: torch.Tensor
    ) -> torch.Tensor:
        """Compute Total Correlation loss."""
        if self.tc == 0:
            return torch.tensor(0.0).to(self.device)
        return self.tc * self._betatc_compute_total_correlation(q_z, q_mean, q_log_var)

    def _compute_mmd_loss(self, q_z: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy loss."""
        if self.info == 0:
            return torch.tensor(0.0).to(self.device)
        prior_samples = torch.randn_like(q_z)
        return self.info * self._compute_mmd(q_z, prior_samples)

    # Legacy API compatibility - keeping original method names
    def take_latent(self, state: np.ndarray) -> np.ndarray:
        """Legacy method name - use extract_latent_representations instead."""
        return self.extract_latent_representations(state)
    
    def take_iembed(self, state: np.ndarray) -> np.ndarray:
        """Legacy method name - use extract_interpretable_embeddings instead."""
        return self.extract_interpretable_embeddings(state)
        
    def take_time(self, state: np.ndarray) -> np.ndarray:
        """Legacy method name - use extract_pseudotime instead."""
        return self.extract_pseudotime(state)
        
    def take_grad(self, state: np.ndarray) -> np.ndarray:
        """Legacy method name - use extract_velocity_gradients instead."""
        return self.extract_velocity_gradients(state)
        
    def take_transition(self, state: np.ndarray, top_k: int = 30) -> np.ndarray:
        """Legacy method name - use compute_transition_probabilities instead."""
        return self.compute_transition_probabilities(state, top_k)
        
    def update(self, states: np.ndarray) -> None:
        """Legacy method name - use train_step instead."""
        self.train_step(states)
