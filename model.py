
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from typing import Tuple, Optional

from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE


class iAODEVAE(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    Integrated Adaptive Ordinary Differential Equation Variational Autoencoder.
    
    This class combines multiple regularization techniques including:
    - Disentangled Information Processing (DIP)
    - Beta-Total Correlation (β-TC)
    - Information bottleneck constraints
    - Optional ODE integration for temporal dynamics
    
    Parameters
    ----------
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
    state_dim : int
        Dimension of the input state space
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
        loss_mode: str,
        lr: float,
        vae_reg: float,
        ode_reg: float,
        device: torch.device,
        *args,
        **kwargs,
    ):
        # Store configuration parameters
        self.use_ode = use_ode
        self.loss_mode = loss_mode
        self.recon = recon
        self.irecon = irecon
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        self.vae_reg = vae_reg
        self.ode_reg = ode_reg
        self.device = device
        
        # Initialize the VAE network
        self.nn = VAE(
            state_dim, hidden_dim, latent_dim, i_dim, use_ode, loss_mode, device
        )
        
        # Initialize optimizer
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        
        # Initialize loss tracking
        self.loss = []

    @torch.no_grad()
    def take_latent(self, state: np.ndarray) -> np.ndarray:
        """
        Extract latent representations from input states.
        
        Parameters
        ----------
        state : np.ndarray
            Input state data
            
        Returns
        -------
        np.ndarray
            Latent representations
        """
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        if self.use_ode:
            q_z, q_m, q_s, t = self.nn.encoder(state)
            t = t.cpu()
            
            # Sort by time and solve ODE
            t_sorted, sort_idx, sort_idxr = np.unique(
                t, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]
            
            # Combine VAE and ODE representations
            return (self.vae_reg * q_z + self.ode_reg * q_z_ode).cpu().numpy()
        else:
            q_z, q_m, q_s = self.nn.encoder(state)
            return q_z.cpu().numpy()

    @torch.no_grad()
    def take_iembed(self, state: np.ndarray) -> np.ndarray:
        """
        Extract information bottleneck embeddings from input states.
        
        Parameters
        ----------
        state : np.ndarray
            Input state data
            
        Returns
        -------
        np.ndarray
            Information bottleneck embeddings
        """
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        
        if self.use_ode:
            q_z, q_m, q_s, t = self.nn.encoder(states)
            t = t.cpu()
            
            # Sort by time and solve ODE
            t_sorted, sort_idx, sort_idxr = np.unique(
                t, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]

            # Apply information bottleneck encoding
            le = self.nn.latent_encoder(q_z)
            le_ode = self.nn.latent_encoder(q_z_ode)
            return (self.vae_reg * le + self.ode_reg * le_ode).cpu().numpy()
        else:
            if self.loss_mode == "zinb":
                q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl = (
                    self.nn(states)
                )
                return le.cpu().numpy()
            else:
                q_z, q_m, q_s, pred_x, le, pred_xl = self.nn(states)
                return le.cpu().numpy()

    @torch.no_grad()
    def take_time(self, state: np.ndarray) -> np.ndarray:
        """
        Extract time parameters from input states (ODE mode only).
        
        Parameters
        ----------
        state : np.ndarray
            Input state data
            
        Returns
        -------
        np.ndarray
            Predicted time parameters
        """
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        _, _, _, t = self.nn.encoder(states)
        return t.detach().cpu().numpy()

    @torch.no_grad()
    def take_grad(self, state: np.ndarray) -> np.ndarray:
        """
        Compute gradients from the ODE solver.
        
        Parameters
        ----------
        state : np.ndarray
            Input state data
            
        Returns
        -------
        np.ndarray
            Computed gradients
        """
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, t = self.nn.encoder(states)
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        return grads

    @torch.no_grad()
    def take_transition(self, state: np.ndarray, top_k: int = 30) -> np.ndarray:
        """
        Compute transition probability matrix based on latent dynamics.
        
        Parameters
        ----------
        state : np.ndarray
            Input state data
        top_k : int, optional
            Number of top transitions to keep for sparsification. Default is 30.
            
        Returns
        -------
        np.ndarray
            Sparse transition probability matrix
        """
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, t = self.nn.encoder(states)
        
        # Compute gradients and future states
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        z_latent = q_z.cpu().numpy()
        z_future = z_latent + 1e-2 * grads
        
        # Compute similarity-based transition probabilities
        distances = pairwise_distances(z_latent, z_future)
        sigma = np.median(distances)
        similarity = np.exp(-(distances**2) / (2 * sigma**2))
        transition_matrix = similarity / similarity.sum(axis=1, keepdims=True)

        def sparsify_transitions(trans_matrix: np.ndarray, top_k: int = top_k) -> np.ndarray:
            """
            Sparsify transition matrix by keeping only top-k transitions per cell.
            
            Parameters
            ----------
            trans_matrix : np.ndarray
                Full transition matrix
            top_k : int
                Number of top transitions to retain
                
            Returns
            -------
            np.ndarray
                Sparsified transition matrix
            """
            n_cells = trans_matrix.shape[0]
            sparse_trans = np.zeros_like(trans_matrix)
            
            for i in range(n_cells):
                top_indices = np.argsort(trans_matrix[i])[::-1][:top_k]
                sparse_trans[i, top_indices] = trans_matrix[i, top_indices]
                sparse_trans[i] /= sparse_trans[i].sum()
                
            return sparse_trans

        transition_matrix = sparsify_transitions(transition_matrix)
        return transition_matrix

    def update(self, states: np.ndarray) -> None:
        """
        Perform one training step with the given states.
        
        Parameters
        ----------
        states : np.ndarray
            Batch of input states for training
        """
        states = torch.tensor(states, dtype=torch.float).to(self.device)

        if self.use_ode:
            # ODE mode forward pass
            if self.loss_mode == "zinb":
                (
                    q_z, q_m, q_s, x, pred_x, dropout_logits, le, le_ode,
                    pred_xl, dropout_logitsl, q_z_ode, pred_x_ode,
                    dropout_logits_ode, pred_xl_ode, dropout_logitsl_ode,
                ) = self.nn(states)
                
                # Compute ODE consistency loss
                qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()

                # Compute ZINB reconstruction loss
                l = x.sum(-1).view(-1, 1)
                pred_x = pred_x * l
                pred_x_ode = pred_x_ode * l
                disp = torch.exp(self.nn.decoder.disp)
                
                recon_loss = (
                    -self._log_zinb(x, pred_x, disp, dropout_logits).sum(-1).mean()
                )
                recon_loss += (
                    -self._log_zinb(x, pred_x_ode, disp, dropout_logits_ode)
                    .sum(-1).mean()
                )

                # Compute information reconstruction loss
                if self.irecon:
                    pred_xl = pred_xl * l
                    pred_xl_ode = pred_xl_ode * l
                    irecon_loss = (
                        -self.irecon * self._log_zinb(x, pred_xl, disp, dropout_logitsl)
                        .sum(-1).mean()
                    )
                    irecon_loss += (
                        -self.irecon * self._log_zinb(x, pred_xl_ode, disp, dropout_logitsl_ode)
                        .sum(-1).mean()
                    )
                else:
                    irecon_loss = torch.zeros(1).to(self.device)

            else:
                (
                    q_z, q_m, q_s, x, pred_x, le, le_ode, pred_xl,
                    q_z_ode, pred_x_ode, pred_xl_ode,
                ) = self.nn(states)
                
                # Compute ODE consistency loss
                qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()

                if self.loss_mode == "nb":
                    # Negative binomial reconstruction loss
                    l = x.sum(-1).view(-1, 1)
                    pred_x = pred_x * l
                    pred_x_ode = pred_x_ode * l
                    disp = torch.exp(self.nn.decoder.disp)
                    
                    recon_loss = -self._log_nb(x, pred_x, disp).sum(-1).mean()
                    recon_loss += -self._log_nb(x, pred_x_ode, disp).sum(-1).mean()

                    if self.irecon:
                        pred_xl = pred_xl * l
                        pred_xl_ode = pred_xl_ode * l
                        irecon_loss = (
                            -self.irecon * self._log_nb(x, pred_xl, disp).sum(-1).mean()
                        )
                        irecon_loss += (
                            -self.irecon * self._log_nb(x, pred_xl_ode, disp).sum(-1).mean()
                        )
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)
                else:
                    # MSE reconstruction loss
                    recon_loss = F.mse_loss(x, pred_x, reduction="none").sum(-1).mean()
                    recon_loss += F.mse_loss(x, pred_x_ode, reduction="none").sum(-1).mean()
                    
                    irecon_loss = F.mse_loss(x, pred_xl, reduction="none").sum(-1).mean()
                    irecon_loss += F.mse_loss(x, pred_xl_ode, reduction="none").sum(-1).mean()

            # Compute KL divergence
            p_m = torch.zeros_like(q_m)
            p_s = torch.zeros_like(q_s)
            kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

            # Compute regularization losses
            dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip else torch.zeros(1).to(self.device)
            tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc else torch.zeros(1).to(self.device)
            mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info else torch.zeros(1).to(self.device)

            # Combine all losses
            total_loss = (
                self.recon * recon_loss + irecon_loss + qz_div + 
                kl_div + dip_loss + tc_loss + mmd_loss
            )

        else:
            # Standard VAE mode forward pass
            if self.loss_mode == "zinb":
                q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl = (
                    self.nn(states)
                )

                # Compute ZINB reconstruction loss
                l = states.sum(-1).view(-1, 1)
                pred_x = pred_x * l
                disp = torch.exp(self.nn.decoder.disp)
                recon_loss = (
                    -self._log_zinb(states, pred_x, disp, dropout_logits).sum(-1).mean()
                )

                # Compute information reconstruction loss
                if self.irecon:
                    pred_xl = pred_xl * l
                    irecon_loss = (
                        -self.irecon * self._log_zinb(states, pred_xl, disp, dropout_logitsl)
                        .sum(-1).mean()
                    )
                else:
                    irecon_loss = torch.zeros(1).to(self.device)

            else:
                q_z, q_m, q_s, pred_x, le, pred_xl = self.nn(states)

                if self.loss_mode == "nb":
                    # Negative binomial reconstruction loss
                    l = states.sum(-1).view(-1, 1)
                    pred_x = pred_x * l
                    disp = torch.exp(self.nn.decoder.disp)
                    recon_loss = -self._log_nb(states, pred_x, disp).sum(-1).mean()

                    if self.irecon:
                        pred_xl = pred_xl * l
                        irecon_loss = (
                            -self.irecon * self._log_nb(states, pred_xl, disp).sum(-1).mean()
                        )
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)
                else:
                    # MSE reconstruction loss
                    recon_loss = F.mse_loss(states, pred_x, reduction="none").sum(-1).mean()
                    irecon_loss = F.mse_loss(states, pred_xl, reduction="none").sum(-1).mean()

            # Compute KL divergence
            p_m = torch.zeros_like(q_m)
            p_s = torch.zeros_like(q_s)
            kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

            # Compute regularization losses
            dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip else torch.zeros(1).to(self.device)
            tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc else torch.zeros(1).to(self.device)
            mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info else torch.zeros(1).to(self.device)

            # Combine all losses
            total_loss = (
                self.recon * recon_loss + irecon_loss + 
                kl_div + dip_loss + tc_loss + mmd_loss
            )

        # Perform optimization step
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        self.nn_optimizer.step()

        # Record losses for tracking
        loss_components = (
            total_loss.item(),
            recon_loss.item(),
            irecon_loss.item(),
            kl_div.item(),
            dip_loss.item(),
            tc_loss.item(),
            mmd_loss.item(),
        )
        self.loss.append(loss_components)

