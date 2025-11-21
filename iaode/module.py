# module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal, Optional
from .mixin import NODEMixin


class Encoder(nn.Module):
    """
    Variational encoder mapping input states to latent distributions.
    
    Supports multiple encoder architectures:
    - 'mlp': Two-layer fully connected network (default)
    - 'mlp_residual': Multi-layer residual MLP
    - 'linear': Single-layer linear encoding
    - 'transformer': TransformerEncoder as feature extraction backbone
    
    Parameters
    ----------
    state_dim : int
        Dimension of input state (number of peaks for scATAC-seq)
    hidden_dim : int
        Dimension of hidden layers
    action_dim : int
        Dimension of latent space
    use_ode : bool, default=False
        Whether to use ODE for trajectory inference
    encoder_type : {'mlp', 'mlp_residual', 'linear', 'transformer'}, default='mlp'
        Type of encoder architecture
    encoder_num_layers : int, default=2
        Number of encoder layers
    encoder_n_heads : int, default=4
        Number of attention heads (for transformer only)
    encoder_d_model : int, optional
        Model dimension for transformer (defaults to hidden_dim if None)
    
    Input Shape
    -----------
    x : torch.Tensor
        (batch_size, state_dim) or (state_dim,)
        Single cells automatically expand to batch_size=1
    
    Output Shape
    ------------
    q_z : torch.Tensor
        Sampled latent vector (batch_size, action_dim)
    q_m : torch.Tensor
        Mean of latent distribution (batch_size, action_dim)
    q_s : torch.Tensor
        Log-variance of latent distribution (batch_size, action_dim)
    t : torch.Tensor, optional
        Inferred pseudotime (batch_size,), only when use_ode=True
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        use_ode: bool = False,
        encoder_type: Literal["mlp", "mlp_residual", "linear", "transformer"] = "mlp",
        encoder_num_layers: int = 2,
        encoder_n_heads: int = 4,
        encoder_d_model: Optional[int] = None,
    ):
        super().__init__()
        self.use_ode = use_ode
        self.encoder_type = encoder_type

        # Build encoder backbone
        if encoder_type == "mlp":
            layers: list[nn.Module] = []
            in_dim = state_dim
            for _ in range(encoder_num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            self.base_network = nn.Sequential(*layers)
            self.out_dim = hidden_dim

        elif encoder_type == "mlp_residual":
            self.input_proj = nn.Linear(state_dim, hidden_dim)
            blocks = []
            for _ in range(encoder_num_layers):
                blocks.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                    )
                )
            self.res_blocks = nn.ModuleList(blocks)
            self.out_dim = hidden_dim

        elif encoder_type == "linear":
            self.base_network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
            )
            self.out_dim = hidden_dim

        elif encoder_type == "transformer":
            if encoder_d_model is None:
                encoder_d_model = hidden_dim
            self.d_model = encoder_d_model

            self.input_proj = nn.Linear(state_dim, encoder_d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=encoder_d_model,
                nhead=encoder_n_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=encoder_num_layers
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.out_dim = encoder_d_model

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Latent distribution parameters
        self.latent_params = nn.Linear(self.out_dim, action_dim * 2)

        # Pseudotime inference head (ODE mode)
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(self.out_dim, 1),
                nn.Sigmoid(),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize network weights using Xavier initialization"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to hidden representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, state_dim)
        
        Returns
        -------
        hidden : torch.Tensor
            Encoded features (batch_size, out_dim)
        """
        # Ensure batch dimension exists
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self.encoder_type in ["mlp", "linear"]:
            hidden = self.base_network(x)

        elif self.encoder_type == "mlp_residual":
            h = self.input_proj(x)
            for block in self.res_blocks:
                h = h + block(h)  # Residual connection
            hidden = h

        elif self.encoder_type == "transformer":
            # Add sequence dimension: (batch, state_dim) -> (batch, 1, state_dim)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x_emb = self.input_proj(x)        # (batch, seq_len, d_model)
            h = self.transformer(x_emb)       # (batch, seq_len, d_model)
            # Pool over sequence dimension
            h = h.transpose(1, 2)             # (batch, d_model, seq_len)
            hidden = self.pool(h).squeeze(-1) # (batch, d_model)

        else:
            raise RuntimeError("Unsupported encoder_type")

        return hidden

    def forward(
        self, x: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass through encoder.
        
        Returns
        -------
        q_z : torch.Tensor
            Sampled latent vector
        q_m : torch.Tensor
            Latent mean
        q_s : torch.Tensor
            Latent log-variance
        t : torch.Tensor, optional
            Inferred pseudotime (only when use_ode=True)
        """
        # Extract features
        hidden = self._encode_features(x)

        # Compute latent distribution parameters
        latent_output = self.latent_params(hidden)
        q_m, q_s = torch.split(latent_output, latent_output.size(-1) // 2, dim=-1)

        # Reparameterization trick
        std = F.softplus(q_s) + 1e-6
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        # Infer pseudotime (ODE mode)
        if self.use_ode:
            t = self.time_encoder(hidden).squeeze(-1)
            return q_z, q_m, q_s, t

        return q_z, q_m, q_s


class Decoder(nn.Module):
    """
    Decoder network mapping latent vectors back to input space.
    
    Supports three loss modes tailored for scATAC-seq data:
    - 'mse': Mean squared error for continuous data
    - 'nb': Negative binomial for count data (recommended)
    - 'zinb': Zero-inflated negative binomial for sparse count data
    
    Parameters
    ----------
    state_dim : int
        Dimension of input space (number of peaks)
    hidden_dim : int
        Dimension of hidden layers
    action_dim : int
        Dimension of latent space
    loss_mode : {'mse', 'nb', 'zinb'}, default='nb'
        Type of reconstruction loss
    
    Notes
    -----
    For scATAC-seq data, 'nb' or 'zinb' is recommended due to the discrete,
    count-based nature of chromatin accessibility measurements.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
    ):
        super().__init__()
        self.loss_mode = loss_mode

        # Shared base network
        self.base_network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Configure output heads based on loss mode
        if loss_mode in ["nb", "zinb"]:
            # Negative binomial: dispersion parameter
            self.disp = nn.Parameter(torch.randn(state_dim))
            # Mean parameter with Softmax normalization
            mean_decoder_seq: nn.Module = nn.Sequential(
                nn.Linear(hidden_dim, state_dim), 
                nn.Softmax(dim=-1)
            )
            self.mean_decoder = mean_decoder_seq
        else:  # 'mse' mode
            self.mean_decoder = nn.Linear(hidden_dim, state_dim)

        # Zero-inflation parameter (ZINB only)
        if loss_mode == "zinb":
            self.dropout_decoder = nn.Linear(hidden_dim, state_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through decoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Latent vector (batch_size, action_dim)
        
        Returns
        -------
        For 'mse' and 'nb' modes:
            mean : torch.Tensor
                Reconstructed output (batch_size, state_dim)
        
        For 'zinb' mode:
            mean : torch.Tensor
                Reconstructed mean (batch_size, state_dim)
            dropout_logits : torch.Tensor
                Zero-inflation logits (batch_size, state_dim)
        """
        hidden = self.base_network(x)
        mean = self.mean_decoder(hidden)

        if self.loss_mode == "zinb":
            dropout_logits = self.dropout_decoder(hidden)
            return mean, dropout_logits

        return mean


class LatentODEfunc(nn.Module):
    """
    Neural ODE function for latent dynamics modeling.
    
    Models continuous temporal dynamics in latent space for trajectory inference
    in single-cell data. The ODE function dx/dt = f(x, t) is parameterized by
    a two-layer neural network.
    
    Parameters
    ----------
    n_latent : int, default=10
        Dimension of latent space
    n_hidden : int, default=25
        Dimension of hidden layer
    
    Notes
    -----
    Used for pseudotime inference in scATAC-seq data, enabling continuous
    trajectory modeling of chromatin accessibility dynamics.
    """

    def __init__(
        self,
        n_latent: int = 10,
        n_hidden: int = 25,
    ):
        super().__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent dynamics gradient.
        
        Parameters
        ----------
        t : torch.Tensor
            Time point
        x : torch.Tensor
            Latent state
        
        Returns
        -------
        dx_dt : torch.Tensor
            Temporal gradient in latent space
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class iVAE(nn.Module, NODEMixin):
    """
    Interpretable Variational Autoencoder (iVAE) with interpretable bottleneck.
    
    Core architecture for iAODE (interpretable Accessibility ODE VAE), designed
    for scATAC-seq data analysis. Combines VAE with information bottleneck for
    interpretable latent representations.
    
    Parameters
    ----------
    state_dim : int
        Dimension of input state (number of peaks)
    hidden_dim : int
        Dimension of hidden layers
    action_dim : int
        Dimension of latent space (full)
    i_dim : int
        Dimension of interpretable bottleneck
    use_ode : bool
        Whether to use Neural ODE for trajectory inference
    loss_mode : {'mse', 'nb', 'zinb'}, default='nb'
        Reconstruction loss type
    encoder_type : {'mlp', 'mlp_residual', 'linear', 'transformer'}, default='mlp'
        Encoder architecture
    encoder_num_layers : int, default=2
        Number of encoder layers
    encoder_n_heads : int, default=4
        Number of attention heads (transformer only)
    encoder_d_model : int, optional
        Transformer model dimension
    device : torch.device
        Device for computation
    
    Notes
    -----
    The interpretable bottleneck (i_dim) provides a compressed representation
    that balances reconstruction quality with biological interpretability.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        i_dim: int,
        use_ode: bool,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        encoder_type: Literal["mlp", "mlp_residual", "linear", "transformer"] = "mlp",
        encoder_num_layers: int = 2,
        encoder_n_heads: int = 4,
        encoder_d_model: Optional[int] = None,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        super().__init__()

        # Initialize encoder
        self.encoder = Encoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            use_ode=use_ode,
            encoder_type=encoder_type,
            encoder_num_layers=encoder_num_layers,
            encoder_n_heads=encoder_n_heads,
            encoder_d_model=encoder_d_model,
        ).to(device)
        
        # Initialize decoder
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode).to(device)

        # Initialize ODE solver
        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim)

        # Interpretable bottleneck layers
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through VAE.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, state_dim)
        
        Returns
        -------
        Tuple containing:
            q_z : torch.Tensor
                Sampled latent vector
            q_m : torch.Tensor
                Latent mean
            q_s : torch.Tensor
                Latent log-variance
            pred_x : torch.Tensor
                Reconstructed input (direct path)
            le : torch.Tensor
                Encoded bottleneck representation
            pred_xl : torch.Tensor
                Reconstructed input (bottleneck path)
            
            Additional returns for ODE mode:
            q_z_ode : torch.Tensor
                ODE-evolved latent
            pred_x_ode : torch.Tensor
                ODE reconstruction
            
            Additional returns for ZINB mode:
            dropout_logits : torch.Tensor
                Zero-inflation parameters
        """
        # Encode
        if self.encoder.use_ode:
            q_z, q_m, q_s, t = self.encoder(x)

            # Sort by pseudotime
            idxs = torch.argsort(t)
            t = t[idxs]
            q_z = q_z[idxs]
            q_m = q_m[idxs]
            q_s = q_s[idxs]
            x = x[idxs]

            # Remove duplicate time points
            unique_mask = torch.ones_like(t, dtype=torch.bool)
            unique_mask[1:] = t[1:] != t[:-1]

            t = t[unique_mask]
            q_z = q_z[unique_mask]
            q_m = q_m[unique_mask]
            q_s = q_s[unique_mask]
            x = x[unique_mask]

            # Solve ODE from initial state
            z0 = q_z[0]
            q_z_ode = self.solve_ode(self.ode_solver, z0, t)
            
            # Information bottleneck paths
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)
            le_ode = self.latent_encoder(q_z_ode)
            ld_ode = self.latent_decoder(le_ode)

            # Decode
            if self.decoder.loss_mode == "zinb":
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                pred_x_ode, dropout_logits_ode = self.decoder(q_z_ode)
                pred_xl_ode, dropout_logitsl_ode = self.decoder(ld_ode)
                return (
                    q_z, q_m, q_s, x,
                    pred_x, dropout_logits,
                    le, le_ode,
                    pred_xl, dropout_logitsl,
                    q_z_ode,
                    pred_x_ode, dropout_logits_ode,
                    pred_xl_ode, dropout_logitsl_ode,
                )
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                pred_x_ode = self.decoder(q_z_ode)
                pred_xl_ode = self.decoder(ld_ode)
                return (
                    q_z, q_m, q_s, x,
                    pred_x,
                    le, le_ode,
                    pred_xl,
                    q_z_ode,
                    pred_x_ode,
                    pred_xl_ode,
                )

        else:
            q_z, q_m, q_s = self.encoder(x)
            
            # Information bottleneck
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            # Decode
            if self.decoder.loss_mode == "zinb":
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                return (
                    q_z, q_m, q_s,
                    pred_x, dropout_logits,
                    le,
                    pred_xl, dropout_logitsl,
                )
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                return (q_z, q_m, q_s, pred_x, le, pred_xl)