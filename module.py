
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal, Optional
from .mixin import NODEMixin


class Encoder(nn.Module):
    """
    Variational encoder network that maps input states to latent distributions.
    
    This encoder supports both standard VAE mode and ODE-enhanced mode where
    additional time parameters are predicted alongside latent distributions.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state space
    hidden_dim : int
        Dimension of the hidden layers in the network
    action_dim : int
        Dimension of the latent space (output dimension)
    use_ode : bool, optional
        Whether to use ODE mode. If True, additional time parameters will be output.
        Default is False.

    Attributes
    ----------
    use_ode : bool
        Flag indicating whether ODE mode is enabled
    base_network : nn.Sequential
        Shared feature extraction network
    latent_params : nn.Linear
        Layer producing mean and log-variance parameters
    time_encoder : nn.Sequential, optional
        Time parameter encoder (only present when use_ode=True)
    """

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        use_ode: bool = False
    ) -> None:
        super().__init__()
        
        if state_dim <= 0 or hidden_dim <= 0 or action_dim <= 0:
            raise ValueError("All dimensions must be positive integers")
            
        self.use_ode = use_ode

        # Shared feature extraction network
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Latent distribution parameters (concatenated mean and log-variance)
        self.latent_params = nn.Linear(hidden_dim, action_dim * 2)

        # Optional time encoder for ODE mode
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Constrain time values to [0, 1]
            )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.01)

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
                - q_z: Sampled latent vector of shape (batch_size, action_dim)
                - q_m: Mean of latent distribution of shape (batch_size, action_dim)
                - q_s: Log variance of latent distribution of shape (batch_size, action_dim)

        If use_ode=True:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                - q_z: Sampled latent vector of shape (batch_size, action_dim)
                - q_m: Mean of latent distribution of shape (batch_size, action_dim)
                - q_s: Log variance of latent distribution of shape (batch_size, action_dim)
                - t: Predicted time parameter of shape (batch_size,)
        
        Raises
        ------
        ValueError
            If input tensor has incorrect shape
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
        
        # Extract features through shared network
        hidden = self.base_network(x)

        # Compute latent distribution parameters
        latent_params = self.latent_params(hidden)
        q_m, q_s = torch.chunk(latent_params, 2, dim=-1)

        # Ensure numerical stability for standard deviation
        std = F.softplus(q_s) + 1e-6

        # Sample from the latent distribution using reparameterization trick
        latent_dist = Normal(q_m, std)
        q_z = latent_dist.rsample()

        if self.use_ode:
            # Predict time parameters for ODE integration
            t = self.time_encoder(hidden).squeeze(-1)
            return q_z, q_m, q_s, t

        return q_z, q_m, q_s


class Decoder(nn.Module):
    """
    Decoder network that reconstructs data from latent representations.

    Supports multiple probabilistic output modes:
    - 'mse': Gaussian distribution (continuous data)
    - 'nb': Negative binomial distribution (count data)
    - 'zinb': Zero-inflated negative binomial (sparse count data)

    Parameters
    ----------
    state_dim : int
        Dimension of the output/reconstruction space
    hidden_dim : int
        Dimension of the hidden layers
    action_dim : int
        Dimension of the input latent space
    loss_mode : Literal['mse', 'nb', 'zinb'], optional
        Probabilistic model for the output distribution. Default is 'nb'.

    Attributes
    ----------
    loss_mode : str
        Current loss mode setting
    base_network : nn.Sequential
        Shared feature transformation network
    mean_decoder : nn.Module
        Network producing distribution mean parameters
    disp : nn.Parameter, optional
        Overdispersion parameter for negative binomial modes
    dropout_decoder : nn.Linear, optional
        Zero-inflation parameter network (ZINB mode only)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
    ) -> None:
        super().__init__()
        
        if state_dim <= 0 or hidden_dim <= 0 or action_dim <= 0:
            raise ValueError("All dimensions must be positive integers")
        if loss_mode not in ["mse", "nb", "zinb"]:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")
            
        self.loss_mode = loss_mode

        # Shared feature transformation network
        self.base_network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Configure output layers based on probabilistic model
        if loss_mode in ["nb", "zinb"]:
            # Negative binomial requires overdispersion parameter
            self.disp = nn.Parameter(torch.randn(state_dim))
            # Use softmax for proper probability normalization
            self.mean_decoder = nn.Sequential(
                nn.Linear(hidden_dim, state_dim), 
                nn.Softmax(dim=-1)
            )
        else:  # MSE mode - direct Gaussian output
            self.mean_decoder = nn.Linear(hidden_dim, state_dim)

        # Zero-inflation parameters for ZINB mode
        if loss_mode == "zinb":
            self.dropout_decoder = nn.Linear(hidden_dim, state_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.01)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x : torch.Tensor
            Latent representation of shape (batch_size, action_dim)

        Returns
        -------
        For 'mse' and 'nb' modes:
            torch.Tensor
                Reconstructed output of shape (batch_size, state_dim)
        For 'zinb' mode:
            Tuple[torch.Tensor, torch.Tensor]
                - mean: Reconstruction mean of shape (batch_size, state_dim)
                - dropout_logits: Zero-inflation logits of shape (batch_size, state_dim)
        
        Raises
        ------
        ValueError
            If input tensor has incorrect shape
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")

        # Transform latent representation
        hidden = self.base_network(x)

        # Generate mean parameters
        mean = self.mean_decoder(hidden)

        if self.loss_mode == "zinb":
            # Additional zero-inflation parameters
            dropout_logits = self.dropout_decoder(hidden)
            return mean, dropout_logits

        return mean


class LatentODEfunc(nn.Module):
    """
    Neural ODE function for modeling latent space dynamics.
    
    This module defines the derivative function f(t, z) for the ODE dz/dt = f(t, z),
    where z represents the latent state.

    Parameters
    ----------
    n_latent : int, optional
        Dimension of the latent space. Default is 10.
    n_hidden : int, optional
        Dimension of the hidden layer. Default is 25.

    Attributes
    ----------
    elu : nn.ELU
        ELU activation function
    fc1 : nn.Linear
        First linear transformation
    fc2 : nn.Linear
        Output linear transformation
    """

    def __init__(
        self,
        n_latent: int = 10,
        n_hidden: int = 25,
    ) -> None:
        super().__init__()
        
        if n_latent <= 0 or n_hidden <= 0:
            raise ValueError("Latent and hidden dimensions must be positive")
        
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative of the latent state.

        Parameters
        ----------
        t : torch.Tensor
            Current time point (may not be used in autonomous systems)
        x : torch.Tensor
            Current latent state of shape (..., n_latent)

        Returns
        -------
        torch.Tensor
            Time derivative dx/dt of shape (..., n_latent)
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class VAE(nn.Module, NODEMixin):
    """
    Variational Autoencoder with optional Neural ODE integration.

    This implementation combines variational autoencoders with neural ODEs for
    modeling continuous-time latent dynamics. It includes an information bottleneck
    mechanism and supports multiple probabilistic output distributions.

    Parameters
    ----------
    state_dim : int
        Dimension of the input/output data space
    hidden_dim : int
        Dimension of hidden layers in encoder/decoder
    action_dim : int
        Dimension of the latent space
    i_dim : int
        Dimension of the information bottleneck layer
    use_ode : bool
        Whether to enable Neural ODE integration
    loss_mode : Literal["mse", "nb", "zinb"], optional
        Probabilistic output model. Default is "nb".
    device : torch.device, optional
        Computation device. Default is CUDA if available, else CPU.

    Attributes
    ----------
    encoder : Encoder
        Variational encoder network
    decoder : Decoder
        Probabilistic decoder network
    ode_solver : LatentODEfunc, optional
        Neural ODE function (only if use_ode=True)
    latent_encoder : nn.Linear
        Information bottleneck encoder
    latent_decoder : nn.Linear
        Information bottleneck decoder
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        i_dim: int,
        use_ode: bool,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        
        # Validate inputs
        for dim, name in [(state_dim, "state_dim"), (hidden_dim, "hidden_dim"), 
                         (action_dim, "action_dim"), (i_dim, "i_dim")]:
            if dim <= 0:
                raise ValueError(f"{name} must be positive, got {dim}")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core VAE components
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode).to(device)

        # Neural ODE solver for latent dynamics
        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim).to(device)

        # Information bottleneck layers
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)
        
        # Initialize bottleneck weights
        for module in [self.latent_encoder, self.latent_decoder]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def _process_ode_data(
        self, q_z: torch.Tensor, q_m: torch.Tensor, q_s: torch.Tensor, 
        x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Process and sort data for ODE integration, removing time duplicates.
        
        Returns
        -------
        Tuple containing sorted and deduplicated tensors: (q_z, q_m, q_s, x, t)
        """
        # Sort by time for proper ODE integration
        time_indices = torch.argsort(t)
        t_sorted = t[time_indices]
        q_z_sorted = q_z[time_indices]
        q_m_sorted = q_m[time_indices]
        q_s_sorted = q_s[time_indices]
        x_sorted = x[time_indices]

        # Remove duplicate time points to avoid ODE solver issues
        if len(t_sorted) > 1:
            unique_mask = torch.ones_like(t_sorted, dtype=torch.bool)
            unique_mask[1:] = t_sorted[1:] != t_sorted[:-1]
            
            t_unique = t_sorted[unique_mask]
            q_z_unique = q_z_sorted[unique_mask]
            q_m_unique = q_m_sorted[unique_mask] 
            q_s_unique = q_s_sorted[unique_mask]
            x_unique = x_sorted[unique_mask]
        else:
            t_unique, q_z_unique, q_m_unique, q_s_unique, x_unique = \
                t_sorted, q_z_sorted, q_m_sorted, q_s_sorted, x_sorted

        return q_z_unique, q_m_unique, q_s_unique, x_unique, t_unique

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, state_dim)

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Variable-length tuple depending on configuration:
            
            Base outputs (always present):
            - q_z: Sampled latent vectors
            - q_m: Latent distribution means  
            - q_s: Latent distribution log-variances
            - [x]: Input data (reordered in ODE mode)
            - pred_x: Direct reconstruction
            - [dropout_logits]: Zero-inflation params (ZINB mode only)
            - le: Information bottleneck encoding
            - [le_ode]: ODE bottleneck encoding (ODE mode only)
            - pred_xl: Bottleneck reconstruction  
            - [dropout_logitsl]: Bottleneck zero-inflation (ZINB mode only)
            
            Additional ODE outputs:
            - q_z_ode: ODE-integrated latent vectors
            - pred_x_ode: ODE direct reconstruction
            - [dropout_logits_ode]: ODE zero-inflation (ZINB mode only)
            - pred_xl_ode: ODE bottleneck reconstruction
            - [dropout_logitsl_ode]: ODE bottleneck zero-inflation (ZINB mode only)
        """
        # Encode input to latent space
        encoder_output = self.encoder(x)
        
        if self.encoder.use_ode:
            # ODE mode: handle time-dependent latent evolution
            q_z, q_m, q_s, t = encoder_output
            
            # Sort and deduplicate time points
            q_z, q_m, q_s, x, t = self._process_ode_data(q_z, q_m, q_s, x, t)
            
            if len(t) == 0:
                raise RuntimeError("No valid time points after deduplication")

            # Integrate latent dynamics using Neural ODE
            z_initial = q_z[0]
            q_z_ode = self.solve_ode(self.ode_solver, z_initial, t)
            
            # Information bottleneck processing
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)
            le_ode = self.latent_encoder(q_z_ode)
            ld_ode = self.latent_decoder(le_ode)

            # Generate outputs based on loss mode
            if self.decoder.loss_mode == "zinb":
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                pred_x_ode, dropout_logits_ode = self.decoder(q_z_ode)
                pred_xl_ode, dropout_logitsl_ode = self.decoder(ld_ode)
                
                return (
                    q_z, q_m, q_s, x, pred_x, dropout_logits,
                    le, le_ode, pred_xl, dropout_logitsl,
                    q_z_ode, pred_x_ode, dropout_logits_ode,
                    pred_xl_ode, dropout_logitsl_ode,
                )
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                pred_x_ode = self.decoder(q_z_ode)
                pred_xl_ode = self.decoder(ld_ode)
                
                return (
                    q_z, q_m, q_s, x, pred_x,
                    le, le_ode, pred_xl,
                    q_z_ode, pred_x_ode, pred_xl_ode,
                )

        else:
            # Standard VAE mode: no time dependency
            q_z, q_m, q_s = encoder_output
            
            # Information bottleneck processing
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            # Generate outputs based on loss mode
            if self.decoder.loss_mode == "zinb":
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                
                return (
                    q_z, q_m, q_s, pred_x, dropout_logits,
                    le, pred_xl, dropout_logitsl,
                )
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                
                return (q_z, q_m, q_s, pred_x, le, pred_xl)
