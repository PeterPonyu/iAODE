# model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import iVAE


class iaodeModel(nn.Module, scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    iAODE (interpretable Accessibility ODE VAE) core model.
    
    Combines variational autoencoding with Neural ODE for trajectory inference
    in scATAC-seq data. Supports multiple reconstruction losses (MSE/NB/ZINB)
    and disentanglement regularizations (β-VAE, DIP, β-TC, InfoVAE).
    
    Parameters
    ----------
    recon : float
        Reconstruction loss weight
    irecon : float
        Interpretable bottleneck reconstruction weight
    beta : float
        KL divergence weight (β-VAE)
    dip : float
        Disentangled Inferred Prior weight
    tc : float
        Total Correlation weight (β-TC-VAE)
    info : float
        Maximum Mean Discrepancy weight (InfoVAE)
    state_dim : int
        Input dimension (number of peaks)
    hidden_dim : int
        Hidden layer dimension
    latent_dim : int
        Full latent space dimension
    i_dim : int
        Interpretable bottleneck dimension
    use_ode : bool
        Enable Neural ODE for trajectory inference
    loss_mode : {'mse', 'nb', 'zinb'}
        Reconstruction loss type
    lr : float
        Learning rate
    vae_reg : float
        VAE latent regularization weight
    ode_reg : float
        ODE latent regularization weight
    device : torch.device
        Computation device
    encoder_type : {'mlp', 'mlp_residual', 'linear', 'transformer'}
        Encoder architecture
    encoder_num_layers : int
        Number of encoder layers
    encoder_n_heads : int
        Number of attention heads (transformer only)
    encoder_d_model : int, optional
        Transformer model dimension
    """

    def __init__(
        self,
        recon,
        irecon,
        beta,
        dip,
        tc,
        info,
        state_dim,
        hidden_dim,
        latent_dim,
        i_dim,
        use_ode,
        loss_mode,
        lr,
        vae_reg,
        ode_reg,
        device,
        encoder_type,
        encoder_num_layers,
        encoder_n_heads,
        encoder_d_model,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Loss weights
        self.recon = recon
        self.irecon = irecon
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        
        # Model config
        self.use_ode = use_ode
        self.loss_mode = loss_mode
        self.vae_reg = vae_reg
        self.ode_reg = ode_reg
        self.device = device
        self.lr = lr

        # Initialize VAE
        self.nn = iVAE(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            loss_mode=loss_mode,
            encoder_type=encoder_type,
            encoder_num_layers=encoder_num_layers,
            encoder_n_heads=encoder_n_heads,
            encoder_d_model=encoder_d_model,
            device=device,
        )

        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.loss = []

    def _compute_loss_only(self, states):
        """Compute loss without gradient update (for validation)"""
        states = states.to(self.device)

        with torch.no_grad():
            if self.use_ode:
                if self.loss_mode == "zinb":
                    (
                        q_z, q_m, q_s, x,
                        pred_x, dropout_logits,
                        le, le_ode,
                        pred_xl, dropout_logitsl,
                        q_z_ode,
                        pred_x_ode, dropout_logits_ode,
                        pred_xl_ode, dropout_logitsl_ode,
                    ) = self.nn(states)

                    L = x.sum(-1).view(-1, 1)
                    pred_x = pred_x * L
                    pred_x_ode = pred_x_ode * L
                    disp = torch.exp(self.nn.decoder.disp)
                    
                    recon_loss = -self._log_zinb(x, pred_x, disp, dropout_logits).sum(-1).mean()
                    recon_loss += -self._log_zinb(x, pred_x_ode, disp, dropout_logits_ode).sum(-1).mean()

                    if self.irecon:
                        pred_xl = pred_xl * L
                        pred_xl_ode = pred_xl_ode * L
                        irecon_loss = -self.irecon * self._log_zinb(x, pred_xl, disp, dropout_logitsl).sum(-1).mean()
                        irecon_loss += -self.irecon * self._log_zinb(x, pred_xl_ode, disp, dropout_logitsl_ode).sum(-1).mean()
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)

                else:
                    (
                        q_z, q_m, q_s, x,
                        pred_x,
                        le, le_ode,
                        pred_xl,
                        q_z_ode,
                        pred_x_ode,
                        pred_xl_ode,
                    ) = self.nn(states)

                    if self.loss_mode == "nb":
                        L = x.sum(-1).view(-1, 1)
                        pred_x = pred_x * L
                        pred_x_ode = pred_x_ode * L
                        disp = torch.exp(self.nn.decoder.disp)
                        
                        recon_loss = -self._log_nb(x, pred_x, disp).sum(-1).mean()
                        recon_loss += -self._log_nb(x, pred_x_ode, disp).sum(-1).mean()

                        if self.irecon:
                            pred_xl = pred_xl * L
                            pred_xl_ode = pred_xl_ode * L
                            irecon_loss = -self.irecon * self._log_nb(x, pred_xl, disp).sum(-1).mean()
                            irecon_loss += -self.irecon * self._log_nb(x, pred_xl_ode, disp).sum(-1).mean()
                        else:
                            irecon_loss = torch.zeros(1).to(self.device)
                    else:
                        recon_loss = F.mse_loss(x, pred_x, reduction="none").sum(-1).mean()
                        recon_loss += F.mse_loss(x, pred_x_ode, reduction="none").sum(-1).mean()
                        irecon_loss = F.mse_loss(x, pred_xl, reduction="none").sum(-1).mean()
                        irecon_loss += F.mse_loss(x, pred_xl_ode, reduction="none").sum(-1).mean()

                qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()
                
            else:
                if self.loss_mode == "zinb":
                    q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl = self.nn(states)

                    L = states.sum(-1).view(-1, 1)
                    pred_x = pred_x * L
                    disp = torch.exp(self.nn.decoder.disp)
                    recon_loss = -self._log_zinb(states, pred_x, disp, dropout_logits).sum(-1).mean()

                    if self.irecon:
                        pred_xl = pred_xl * L
                        irecon_loss = -self.irecon * self._log_zinb(x, pred_xl, disp, dropout_logitsl).sum(-1).mean()
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)

                else:
                    q_z, q_m, q_s, pred_x, le, pred_xl = self.nn(states)

                    if self.loss_mode == "nb":
                        L = states.sum(-1).view(-1, 1)
                        pred_x = pred_x * L
                        disp = torch.exp(self.nn.decoder.disp)
                        recon_loss = -self._log_nb(states, pred_x, disp).sum(-1).mean()

                        if self.irecon:
                            pred_xl = pred_xl * l
                            irecon_loss = -self.irecon * self._log_nb(states, pred_xl, disp).sum(-1).mean()
                        else:
                            irecon_loss = torch.zeros(1).to(self.device)
                    else:
                        recon_loss = F.mse_loss(states, pred_x, reduction="none").sum(-1).mean()
                        irecon_loss = F.mse_loss(states, pred_xl, reduction="none").sum(-1).mean()

                qz_div = torch.zeros(1).to(self.device)

            # Regularization losses
            p_m = torch.zeros_like(q_m)
            p_s = torch.zeros_like(q_s)
            kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

            dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip else torch.zeros(1).to(self.device)
            tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc else torch.zeros(1).to(self.device)
            mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info else torch.zeros(1).to(self.device)

            total_loss = self.recon * recon_loss + irecon_loss + qz_div + kl_div + dip_loss + tc_loss + mmd_loss

        return total_loss.item()

    @torch.no_grad()
    def take_latent(self, state):
        """Extract latent representation (ODE-regularized if enabled)"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        else:
            state = state.to(self.device)

        if self.use_ode:
            q_z, q_m, q_s, t = self.nn.encoder(state)
            t = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(t, return_index=True, return_inverse=True)
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]
            return (self.vae_reg * q_z + self.ode_reg * q_z_ode).cpu().numpy()
        else:
            q_z, q_m, q_s = self.nn.encoder(state)
            return q_z.cpu().numpy()

    @torch.no_grad()
    def take_iembed(self, state):
        """Extract interpretable bottleneck embedding"""
        if not isinstance(state, torch.Tensor):
            states = torch.tensor(state, dtype=torch.float).to(self.device)
        else:
            states = state.to(self.device)

        if self.use_ode:
            q_z, q_m, q_s, t = self.nn.encoder(states)
            t = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(t, return_index=True, return_inverse=True)
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]

            le = self.nn.latent_encoder(q_z)
            le_ode = self.nn.latent_encoder(q_z_ode)
            return (self.vae_reg * le + self.ode_reg * le_ode).cpu().numpy()
        else:
            if self.loss_mode == "zinb":
                q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl = self.nn(states)
            else:
                q_z, q_m, q_s, pred_x, le, pred_xl = self.nn(states)
            return le.cpu().numpy()

    @torch.no_grad()
    def take_time(self, state):
        """Extract inferred pseudotime (ODE mode only)"""
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        _, _, _, t = self.nn.encoder(states)
        return t.detach().cpu().numpy()

    @torch.no_grad()
    def take_grad(self, state):
        """Extract latent dynamics gradient (ODE mode only)"""
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, t = self.nn.encoder(states)
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        return grads

    @torch.no_grad()
    def take_transition(self, state, top_k: int = 30):
        """
        Compute cell-cell transition probability matrix.
        
        Parameters
        ----------
        state : array-like
            Input cell states
        top_k : int, default=30
            Number of nearest neighbors to retain (sparsification)
        
        Returns
        -------
        transition_matrix : np.ndarray
            Sparse transition probability matrix (n_cells × n_cells)
        """
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, t = self.nn.encoder(states)
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        z_latent = q_z.cpu().numpy()
        z_future = z_latent + 1e-2 * grads
        
        distances = pairwise_distances(z_latent, z_future)
        sigma = np.median(distances)
        similarity = np.exp(-(distances**2) / (2 * sigma**2))
        transition_matrix = similarity / similarity.sum(axis=1, keepdims=True)

        # Sparsify to top-k transitions
        n_cells = transition_matrix.shape[0]
        sparse_trans = np.zeros_like(transition_matrix)
        for i in range(n_cells):
            top_indices = np.argsort(transition_matrix[i])[::-1][:top_k]
            sparse_trans[i, top_indices] = transition_matrix[i, top_indices]
            sparse_trans[i] /= sparse_trans[i].sum()

        return sparse_trans

    def update(self, states):
        """Perform one gradient update step"""
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float).to(self.device)
        else:
            states = states.to(self.device)

        # Forward pass
        if self.use_ode:
            if self.loss_mode == "zinb":
                (
                    q_z, q_m, q_s, x,
                    pred_x, dropout_logits,
                    le, le_ode,
                    pred_xl, dropout_logitsl,
                    q_z_ode,
                    pred_x_ode, dropout_logits_ode,
                    pred_xl_ode, dropout_logitsl_ode,
                ) = self.nn(states)
                
                qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()

                L = x.sum(-1).view(-1, 1)
                pred_x = pred_x * L
                pred_x_ode = pred_x_ode * L
                disp = torch.exp(self.nn.decoder.disp)
                recon_loss = -self._log_zinb(x, pred_x, disp, dropout_logits).sum(-1).mean()
                recon_loss += -self._log_zinb(x, pred_x_ode, disp, dropout_logits_ode).sum(-1).mean()

                if self.irecon:
                    pred_xl = pred_xl * L
                    pred_xl_ode = pred_xl_ode * L
                    irecon_loss = -self.irecon * self._log_zinb(x, pred_xl, disp, dropout_logitsl).sum(-1).mean()
                    irecon_loss += -self.irecon * self._log_zinb(x, pred_xl_ode, disp, dropout_logitsl_ode).sum(-1).mean()
                else:
                    irecon_loss = torch.zeros(1).to(self.device)

            else:
                (
                    q_z, q_m, q_s, x,
                    pred_x,
                    le, le_ode,
                    pred_xl,
                    q_z_ode,
                    pred_x_ode,
                    pred_xl_ode,
                ) = self.nn(states)
                
                qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()

                if self.loss_mode == "nb":
                    L = x.sum(-1).view(-1, 1)
                    pred_x = pred_x * L
                    pred_x_ode = pred_x_ode * L
                    disp = torch.exp(self.nn.decoder.disp)
                    recon_loss = -self._log_nb(x, pred_x, disp).sum(-1).mean()
                    recon_loss += -self._log_nb(x, pred_x_ode, disp).sum(-1).mean()

                    if self.irecon:
                        pred_xl = pred_xl * L
                        pred_xl_ode = pred_xl_ode * L
                        irecon_loss = -self.irecon * self._log_nb(x, pred_xl, disp).sum(-1).mean()
                        irecon_loss += -self.irecon * self._log_nb(x, pred_xl_ode, disp).sum(-1).mean()
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)
                else:
                    recon_loss = F.mse_loss(x, pred_x, reduction="none").sum(-1).mean()
                    recon_loss += F.mse_loss(x, pred_x_ode, reduction="none").sum(-1).mean()
                    irecon_loss = F.mse_loss(x, pred_xl, reduction="none").sum(-1).mean()
                    irecon_loss += F.mse_loss(x, pred_xl_ode, reduction="none").sum(-1).mean()

        else:
            if self.loss_mode == "zinb":
                q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl = self.nn(states)

                l = states.sum(-1).view(-1, 1)
                pred_x = pred_x * l
                disp = torch.exp(self.nn.decoder.disp)
                recon_loss = -self._log_zinb(states, pred_x, disp, dropout_logits).sum(-1).mean()

                if self.irecon:
                    pred_xl = pred_xl * l
                    irecon_loss = -self.irecon * self._log_zinb(states, pred_xl, disp, dropout_logitsl).sum(-1).mean()
                else:
                    irecon_loss = torch.zeros(1).to(self.device)

            else:
                q_z, q_m, q_s, pred_x, le, pred_xl = self.nn(states)

                if self.loss_mode == "nb":
                    l = states.sum(-1).view(-1, 1)
                    pred_x = pred_x * l
                    disp = torch.exp(self.nn.decoder.disp)
                    recon_loss = -self._log_nb(states, pred_x, disp).sum(-1).mean()

                    if self.irecon:
                        pred_xl = pred_xl * l
                        irecon_loss = -self.irecon * self._log_nb(states, pred_xl, disp).sum(-1).mean()
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)
                else:
                    recon_loss = F.mse_loss(states, pred_x, reduction="none").sum(-1).mean()
                    irecon_loss = F.mse_loss(states, pred_xl, reduction="none").sum(-1).mean()

            qz_div = torch.zeros(1).to(self.device)

        # Regularization losses
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip else torch.zeros(1).to(self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc else torch.zeros(1).to(self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info else torch.zeros(1).to(self.device)

        # Total loss
        total_loss = (
            self.recon * recon_loss
            + irecon_loss
            + qz_div
            + kl_div
            + dip_loss
            + tc_loss
            + mmd_loss
        )

        # Backpropagation
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        self.nn_optimizer.step()

        # Record losses
        self.loss.append((
            total_loss.item(),
            recon_loss.item(),
            irecon_loss.item(),
            kl_div.item(),
            dip_loss.item(),
            tc_loss.item(),
            mmd_loss.item(),
        ))