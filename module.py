#module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal, Optional
from .mixin import NODEMixin


class Encoder(nn.Module):
    """
    变分编码器网络，将输入状态映射到潜在分布。

    支持多种编码器结构:
    - 'mlp': 两层全连接网络（默认）
    - 'mlp_residual': 多层残差 MLP
    - 'linear': 单层线性编码
    - 'transformer': 使用 TransformerEncoder 作为特征提取 backbone

    输入:
        x: 一般为 (batch_size, state_dim) 的张量。
            若为 (state_dim,), 会自动视为 batch_size=1。
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

        # ---------- choose backbone ----------
        if encoder_type == "mlp":
            layers = []
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
                batch_first=True,  # (batch, seq_len, d_model)
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=encoder_num_layers
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.out_dim = encoder_d_model

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # ---------- latent parameters ----------
        self.latent_params = nn.Linear(self.out_dim, action_dim * 2)

        # time head for ODE mode
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(self.out_dim, 1),
                nn.Sigmoid(),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入 x 编码为 hidden 表示。
        输入:
            x: 至少 2 维, (batch_size, state_dim)
        输出:
            hidden: (batch_size, out_dim)
        """
        # 确保存在 batch 维度 (batch_size, state_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self.encoder_type in ["mlp", "linear"]:
            hidden = self.base_network(x)  # (batch, out_dim)

        elif self.encoder_type == "mlp_residual":
            h = self.input_proj(x)  # (batch, hidden_dim)
            for block in self.res_blocks:
                h = h + block(h)     # residual
            hidden = h              # (batch, hidden_dim)

        elif self.encoder_type == "transformer":
            # (batch, state_dim) -> (batch, 1, state_dim)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            # 若未来你改成真正的序列输入，x 可以是 (batch, seq_len, state_dim)
            x_emb = self.input_proj(x)        # (batch, seq_len, d_model)
            h = self.transformer(x_emb)       # (batch, seq_len, d_model)
            # pool over seq_len -> (batch, d_model)
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
        # 1) 提取特征 (batch, out_dim)
        hidden = self._encode_features(x)

        # 2) 计算潜在分布参数
        latent_output = self.latent_params(hidden)   # (batch, 2 * action_dim)
        q_m, q_s = torch.split(latent_output, latent_output.size(-1) // 2, dim=-1)

        std = F.softplus(q_s) + 1e-6
        dist = Normal(q_m, std)
        q_z = dist.rsample()                         # (batch, action_dim)

        # 3) ODE 模式下输出时间 t
        if self.use_ode:
            t = self.time_encoder(hidden).squeeze(-1)  # (batch,)
            return q_z, q_m, q_s, t

        return q_z, q_m, q_s


class Decoder(nn.Module):
    """
    解码器网络，将潜在向量映射回原始空间

    支持三种损失模式：
    - 'mse': 均方误差损失，适用于连续数据
    - 'nb': 负二项分布损失，适用于离散计数数据
    - 'zinb': 零膨胀负二项分布损失，适用于有大量零值的计数数据

    参数
    ----------
    state_dim : int
        原始空间的维度
    hidden_dim : int
        隐藏层的维度
    action_dim : int
        潜在空间的维度
    loss_mode : Literal['mse', 'nb', 'zinb']
        损失函数的模式，默认为'nb'
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

        # 共享基础网络部分
        self.base_network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 根据损失模式配置输出层
        if loss_mode in ["nb", "zinb"]:
            # 负二项分布参数：过离散参数
            self.disp = nn.Parameter(torch.randn(state_dim))
            # 均值参数：使用Softmax确保归一化
            self.mean_decoder = nn.Sequential(
                nn.Linear(hidden_dim, state_dim), nn.Softmax(dim=-1)
            )
        else:  # 'mse'模式
            # 直接线性输出
            self.mean_decoder = nn.Linear(hidden_dim, state_dim)

        # 零膨胀参数 (仅用于'zinb'模式)
        if loss_mode == "zinb":
            self.dropout_decoder = nn.Linear(hidden_dim, state_dim)

        # 应用权重初始化
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """初始化网络权重"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        参数
        ----------
        x : torch.Tensor
            形状为(batch_size, action_dim)的潜在向量

        返回
        ----------
        对于'mse'和'nb'模式：
            torch.Tensor: 重构的输出
        对于'zinb'模式：
            Tuple[torch.Tensor, torch.Tensor]: (重构均值, 零膨胀参数的logits)
        """
        # 通过基础网络
        hidden = self.base_network(x)

        # 计算均值输出
        mean = self.mean_decoder(hidden)

        # 对于'zinb'模式，还需计算零膨胀参数
        if self.loss_mode == "zinb":
            dropout_logits = self.dropout_decoder(hidden)
            return mean, dropout_logits

        # 对于'mse'和'nb'模式，只返回均值
        return mean


class LatentODEfunc(nn.Module):
    """
    潜在空间ODE函数模型

    参数:
    n_latent: 潜在空间维度
    n_hidden: 隐藏层维度
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
        计算在时间t和状态x下的梯度

        参数:
        t: 时间点
        x: 潜在状态

        返回:
        梯度值
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class VAE(nn.Module, NODEMixin):
    """
    Variational Autoencoder with support for both linear.

    Parameters
    ----------
    state_dim : int
        Dimension of input state space
    hidden_dim : int
        Dimension of hidden layers
    action_dim : int
        Dimension of action/latent space
    i_dim : int
        Dimension of information bottleneck

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
        
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode).to(device)

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
            - q_z: sampled latent vector
            - q_m: mean of encoding distribution
            - q_s: log variance of encoding distribution
            - pred_x: reconstructed input (direct path)
            - le: encoded information bottleneck
            - ld: decoded information bottleneck
            - pred_xl: reconstructed input (bottleneck path)
        """
        # Encode
        if self.encoder.use_ode:
            q_z, q_m, q_s, t = self.encoder(x)

            idxs = torch.argsort(t)
            t = t[idxs]
            q_z = q_z[idxs]
            q_m = q_m[idxs]
            q_s = q_s[idxs]
            x = x[idxs]

            unique_mask = torch.ones_like(t, dtype=torch.bool)
            unique_mask[1:] = t[1:] != t[:-1]

            t = t[unique_mask]
            q_z = q_z[unique_mask]
            q_m = q_m[unique_mask]
            q_s = q_s[unique_mask]
            x = x[unique_mask]

            z0 = q_z[0]
            q_z_ode = self.solve_ode(self.ode_solver, z0, t)
            # Information bottleneck
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            le_ode = self.latent_encoder(q_z_ode)
            ld_ode = self.latent_decoder(le_ode)

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
            q_z, q_m, q_s = self.encoder(x)
            # Information bottleneck
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

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