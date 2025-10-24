#module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal
from .mixin import NODEMixin


class Encoder(nn.Module):
    """
    变分编码器网络，将输入状态映射到潜在分布。

    参数
    ----------
    state_dim : int
        输入状态的维度
    hidden_dim : int
        隐藏层的维度
    action_dim : int
        潜在空间的维度
    use_ode : bool
        是否使用ODE模式，若为True则会额外输出时间参数
    """

    def __init__(
        self, state_dim: int, hidden_dim: int, action_dim: int, use_ode: bool = False
    ):
        super().__init__()
        self.use_ode = use_ode

        # 基础网络部分
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 潜在空间参数（均值和对数方差）
        self.latent_params = nn.Linear(hidden_dim, action_dim * 2)

        # 时间编码器（仅在ODE模式下使用）
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # 使用Sigmoid确保时间值在0-1范围内
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """初始化网络权重"""
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
        编码器的前向传播

        参数
        ----------
        x : torch.Tensor
            形状为(batch_size, state_dim)的输入张量

        返回
        -------
        如果use_ode=False:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                - 采样的潜在向量
                - 潜在分布的均值
                - 潜在分布的对数方差

        如果use_ode=True:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                - 采样的潜在向量
                - 潜在分布的均值
                - 潜在分布的对数方差
                - 预测的时间参数 (0-1范围内)
        """
        # 通过基础网络获取特征
        hidden = self.base_network(x)

        # 计算潜在空间参数
        latent_output = self.latent_params(hidden)
        q_m, q_s = torch.split(latent_output, latent_output.size(-1) // 2, dim=-1)

        # 使用softplus确保方差为正
        std = F.softplus(q_s) + 1e-6

        # 从分布中采样
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        # 如果使用ODE模式，额外输出时间参数
        if self.use_ode:
            t = self.time_encoder(hidden).squeeze(
                -1
            )  # 移除最后一个维度使t为(batch_size,)
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
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        super().__init__()

        # Initialize encoder
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(device)
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