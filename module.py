
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal
from .mixin import NODEMixin


class Encoder(nn.Module):
    """
    Variational encoder network that maps input states to latent distributions.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state
    hidden_dim : int
        Dimension of the hidden layers
    action_dim : int
        Dimension of the latent space
    use_ode : bool, optional
        Whether to use ODE mode. If True, additional time parameters will be output.
        Default is False.
    """

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        use_ode: bool = False
    ):
        super().__init__()
        self.use_ode = use_ode

        # Base network layers
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent space parameters (mean and log variance)
        self.latent_params = nn.Linear(hidden_dim, action_dim * 2)

        # Time encoder (only used in ODE mode)
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Ensure time values are in the range [0, 1]
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize network weights using Xavier normal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def forward(
        self, x: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, state_dim)

        Returns
        -------
        If use_ode=False:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                - Sampled latent vector
                - Mean of the latent distribution  
                - Log variance of the latent distribution

        If use_ode=True:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                - Sampled latent vector
                - Mean of the latent distribution
                - Log variance of the latent distribution
                - Predicted time parameter (in range [0, 1])
        """
        # Extract features through base network
        hidden = self.base_network(x)

        # Compute latent space parameters
        latent_output = self.latent_params(hidden)
        q_m, q_s = torch.split(latent_output, latent_output.size(-1) // 2, dim=-1)

        # Ensure positive variance using softplus
        std = F.softplus(q_s) + 1e-6

        # Sample from the distribution
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        # If using ODE mode, output additional time parameters
        if self.use_ode:
            t = self.time_encoder(hidden).squeeze(-1)  # Remove last dimension to make t (batch_size,)
            return q_z, q_m, q_s, t

        return q_z, q_m, q_s


class Decoder(nn.Module):
    """
    Decoder network that maps latent vectors back to the original space.

    Supports three loss modes:
    - 'mse': Mean squared error loss, suitable for continuous data
    - 'nb': Negative binomial distribution loss, suitable for discrete count data
    - 'zinb': Zero-inflated negative binomial distribution loss, suitable for count data with many zeros

    Parameters
    ----------
    state_dim : int
        Dimension of the original space
    hidden_dim : int
        Dimension of the hidden layers
    action_dim : int
        Dimension of the latent space
    loss_mode : Literal['mse', 'nb', 'zinb'], optional
        Mode of the loss function. Default is 'nb'.
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

        # Shared base network layers
        self.base_network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Configure output layers based on loss mode
        if loss_mode in ["nb", "zinb"]:
            # Negative binomial distribution parameters: overdispersion parameter
            self.disp = nn.Parameter(torch.randn(state_dim))
            # Mean parameter: use Softmax to ensure normalization
            self.mean_decoder = nn.Sequential(
                nn.Linear(hidden_dim, state_dim), 
                nn.Softmax(dim=-1)
            )
        else:  # 'mse' mode
            # Direct linear output
            self.mean_decoder = nn.Linear(hidden_dim, state_dim)

        # Zero-inflation parameter (only used for 'zinb' mode)
        if loss_mode == "zinb":
            self.dropout_decoder = nn.Linear(hidden_dim, state_dim)

        # Apply weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize network weights using Xavier normal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x : torch.Tensor
            Latent vector of shape (batch_size, action_dim)

        Returns
        -------
        For 'mse' and 'nb' modes:
            torch.Tensor
                Reconstructed output
        For 'zinb' mode:
            Tuple[torch.Tensor, torch.Tensor]
                (Reconstructed mean, Zero-inflation parameter logits)
        """
        # Pass through base network
        hidden = self.base_network(x)

        # Compute mean output
        mean = self.mean_decoder(hidden)

        # For 'zinb' mode, also compute zero-inflation parameters
        if self.loss_mode == "zinb":
            dropout_logits = self.dropout_decoder(hidden)
            return mean, dropout_logits

        # For 'mse' and 'nb' modes, return only the mean
        return mean


class LatentODEfunc(nn.Module):
    """
    Latent space ODE function model.

    Parameters
    ----------
    n_latent : int, optional
        Dimension of the latent space. Default is 10.
    n_hidden : int, optional
        Dimension of the hidden layers. Default is 25.
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
        Compute the gradient at time t and state x.

        Parameters
        ----------
        t : torch.Tensor
            Time point
        x : torch.Tensor
            Latent state

        Returns
        -------
        torch.Tensor
            Gradient values
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class VAE(nn.Module, NODEMixin):
    """
    Variational Autoencoder with support for both linear and ODE modes.

    This implementation includes an information bottleneck mechanism and supports
    multiple loss modes for different types of data distributions.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state space
    hidden_dim : int
        Dimension of the hidden layers
    action_dim : int
        Dimension of the action/latent space
    i_dim : int
        Dimension of the information bottleneck
    use_ode : bool
        Whether to use ODE integration for latent dynamics
    loss_mode : Literal["mse", "nb", "zinb"], optional
        Loss function mode. Default is "nb".
    device : torch.device, optional
        Device to run the model on. Default is CUDA if available, otherwise CPU.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        i_dim: int,
        use_ode: bool,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        super().__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode).to(device)

        # Initialize ODE solver if needed
        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim)

        # Information bottleneck layers
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, ...]
            The return tuple varies based on use_ode and loss_mode settings:
            - q_z: sampled latent vector
            - q_m: mean of encoding distribution
            - q_s: log variance of encoding distribution
            - pred_x: reconstructed input (direct path)
            - le: encoded information bottleneck
            - ld: decoded information bottleneck
            - pred_xl: reconstructed input (bottleneck path)
            Additional outputs for ODE mode and ZINB loss mode are included when applicable.
        """
        # Encode input
        if self.encoder.use_ode:
            q_z, q_m, q_s, t = self.encoder(x)

            # Sort by time for ODE integration
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

            # Solve ODE starting from initial latent state
            z0 = q_z[0]
            q_z_ode = self.solve_ode(self.ode_solver, z0, t)
            
            # Information bottleneck processing
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            le_ode = self.latent_encoder(q_z_ode)
            ld_ode = self.latent_decoder(le_ode)

            # Decode based on loss mode
            if self.decoder.loss_mode == "zinb":
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                pred_x_ode, dropout_logits_ode = self.decoder(q_z_ode)
                pred_xl_ode, dropout_logitsl_ode = self.decoder(ld_ode)
                return (
                    q_z,
                    q_m,
                    q_s,
                    x,
                    pred_x,
                    dropout_logits,
                    le,
                    le_ode,
                    pred_xl,
                    dropout_logitsl,
                    q_z_ode,
                    pred_x_ode,
                    dropout_logits_ode,
                    pred_xl_ode,
                    dropout_logitsl_ode,
                )
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                pred_x_ode = self.decoder(q_z_ode)
                pred_xl_ode = self.decoder(ld_ode)
                return (
                    q_z,
                    q_m,
                    q_s,
                    x,
                    pred_x,
                    le,
                    le_ode,
                    pred_xl,
                    q_z_ode,
                    pred_x_ode,
                    pred_xl_ode,
                )

        else:
            # Standard VAE forward pass without ODE
            q_z, q_m, q_s = self.encoder(x)
            
            # Information bottleneck processing
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            # Decode based on loss mode
            if self.decoder.loss_mode == "zinb":
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                return (
                    q_z,
                    q_m,
                    q_s,
                    pred_x,
                    dropout_logits,
                    le,
                    pred_xl,
                    dropout_logitsl,
                )
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                return (q_z, q_m, q_s, pred_x, le, pred_xl)

