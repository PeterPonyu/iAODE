
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from typing import Optional, Tuple, Dict, Any
import math


class scviMixin:
    """
    Mixin class providing scVI-style probabilistic distribution utilities.
    
    This mixin implements various probability distribution functions commonly used
    in single-cell variational inference, including KL divergence computations
    and log-likelihood calculations for negative binomial and zero-inflated 
    negative binomial distributions.
    """

    def _normal_kl(
        self, 
        mu1: torch.Tensor, 
        lv1: torch.Tensor, 
        mu2: torch.Tensor, 
        lv2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between two normal distributions.

        Parameters
        ----------
        mu1 : torch.Tensor
            Mean of the first distribution
        lv1 : torch.Tensor
            Log variance of the first distribution
        mu2 : torch.Tensor
            Mean of the second distribution
        lv2 : torch.Tensor
            Log variance of the second distribution

        Returns
        -------
        torch.Tensor
            KL divergence values
        """
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.0
        lstd2 = lv2 / 2.0
        
        kl = lstd2 - lstd1 + (v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2) - 0.5
        return kl

    def _log_nb(
        self, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        theta: torch.Tensor, 
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute log probability under negative binomial distribution.

        Parameters
        ----------
        x : torch.Tensor
            Observed data
        mu : torch.Tensor
            Distribution mean parameter
        theta : torch.Tensor
            Dispersion parameter
        eps : float, optional
            Small constant for numerical stability. Default is 1e-8.

        Returns
        -------
        torch.Tensor
            Log probability values
        """
        log_theta_mu_eps = torch.log(theta + mu + eps)
        
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        return res

    def _log_zinb(
        self, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        theta: torch.Tensor, 
        pi: torch.Tensor, 
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute log probability under zero-inflated negative binomial distribution.

        Parameters
        ----------
        x : torch.Tensor
            Observed data
        mu : torch.Tensor
            Distribution mean parameter
        theta : torch.Tensor
            Dispersion parameter
        pi : torch.Tensor
            Zero-inflation mixing weight logits
        eps : float, optional
            Small constant for numerical stability. Default is 1e-8.

        Returns
        -------
        torch.Tensor
            Log probability values
        """
        softplus_pi = F.softplus(-pi)
        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

        # Case when x == 0 (zero-inflated component)
        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

        # Case when x > 0 (negative binomial component)
        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

        res = mul_case_zero + mul_case_non_zero
        return res


class NODEMixin:
    """
    Mixin class providing Neural Ordinary Differential Equation (NODE) functionality.
    
    This mixin implements utilities for solving ODEs using the torchdiffeq library,
    including step size configuration and device-aware ODE solving.
    """

    @staticmethod
    def get_step_size(
        step_size: Optional[str], 
        t0: float, 
        t1: float, 
        n_points: int
    ) -> Dict[str, Any]:
        """
        Configure step size for ODE solver.

        Parameters
        ----------
        step_size : Optional[str]
            Step size specification. If None, uses adaptive step size.
            If "auto", computes step size automatically.
        t0 : float
            Start time
        t1 : float
            End time
        n_points : int
            Number of time points

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary for ODE solver
        """
        if step_size is None:
            return {}
        else:
            if step_size == "auto":
                step_size = (t1 - t0) / (n_points - 1)
            return {"step_size": step_size}

    def solve_ode(
        self,
        ode_func: torch.nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        method: str = "rk4",
        step_size: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Solve ODE using torchdiffeq.

        This method handles device compatibility by moving tensors to CPU for
        ODE solving and then back to the original device.

        Parameters
        ----------
        ode_func : torch.nn.Module
            ODE function model that defines dx/dt = f(t, x)
        z0 : torch.Tensor
            Initial state vector
        t : torch.Tensor
            Time points at which to evaluate the solution
        method : str, optional
            ODE solving method. Default is "rk4".
        step_size : Optional[float], optional
            Fixed step size for the solver. Default is None (adaptive).

        Returns
        -------
        torch.Tensor
            ODE solution at the specified time points
        """
        options = self.get_step_size(step_size, t[0], t[-1], len(t))
        pred_z = odeint(ode_func, z0, t, method=method, options=options)
        return pred_z


class betatcMixin:
    """
    Mixin class providing β-TC VAE (Beta Total Correlation VAE) functionality.
    
    This mixin implements the total correlation computation used in β-TC VAE
    for disentangled representation learning.
    """

    def _betatc_compute_gaussian_log_density(
        self, 
        samples: torch.Tensor, 
        mean: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log density of Gaussian distribution.

        Parameters
        ----------
        samples : torch.Tensor
            Sample points
        mean : torch.Tensor
            Distribution mean
        log_var : torch.Tensor
            Distribution log variance

        Returns
        -------
        torch.Tensor
            Log density values
        """
        pi = torch.tensor(math.pi, requires_grad=False)
        normalization = torch.log(2 * pi)
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

    def _betatc_compute_total_correlation(
        self, 
        z_sampled: torch.Tensor, 
        z_mean: torch.Tensor, 
        z_logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total correlation for β-TC VAE.

        This measures the mutual information between latent dimensions,
        encouraging statistical independence for disentanglement.

        Parameters
        ----------
        z_sampled : torch.Tensor
            Sampled latent variables
        z_mean : torch.Tensor
            Mean of latent distribution
        z_logvar : torch.Tensor
            Log variance of latent distribution

        Returns
        -------
        torch.Tensor
            Total correlation value
        """
        # Compute log q(z|x) for all samples and latent dimensions
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(dim=1),
            z_mean.unsqueeze(dim=0),
            z_logvar.unsqueeze(dim=0),
        )
        
        # Compute log ∏_j q(z_j|x) (product of marginals)
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        
        # Compute log q(z|x) (joint distribution)
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        
        # Total correlation = E[log q(z|x) - log ∏_j q(z_j|x)]
        return (log_qz - log_qz_product).mean()


class infoMixin:
    """
    Mixin class providing information-theoretic loss functions.
    
    This mixin implements Maximum Mean Discrepancy (MMD) computation using
    RBF kernels for measuring distributional differences.
    """

    def _compute_mmd(
        self, 
        z_posterior_samples: torch.Tensor, 
        z_prior_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD) between posterior and prior samples.

        MMD measures the distance between two distributions using kernel embeddings.

        Parameters
        ----------
        z_posterior_samples : torch.Tensor
            Samples from the posterior distribution
        z_prior_samples : torch.Tensor
            Samples from the prior distribution

        Returns
        -------
        torch.Tensor
            MMD value
        """
        # Compute kernel means for all combinations
        mean_pz_pz = self._compute_unbiased_mean(
            self._compute_kernel(z_prior_samples, z_prior_samples), unbaised=True
        )
        mean_pz_qz = self._compute_unbiased_mean(
            self._compute_kernel(z_prior_samples, z_posterior_samples), unbaised=False
        )
        mean_qz_qz = self._compute_unbiased_mean(
            self._compute_kernel(z_posterior_samples, z_posterior_samples), unbaised=True
        )
        
        # MMD² = k(p,p) - 2k(p,q) + k(q,q)
        mmd = mean_pz_pz - 2 * mean_pz_qz + mean_qz_qz
        return mmd

    def _compute_unbiased_mean(
        self, 
        kernel: torch.Tensor, 
        unbaised: bool
    ) -> torch.Tensor:
        """
        Compute unbiased mean of kernel matrix.

        Parameters
        ----------
        kernel : torch.Tensor
            Kernel matrix
        unbaised : bool
            Whether to compute unbiased estimator (excludes diagonal terms)

        Returns
        -------
        torch.Tensor
            Mean kernel value
        """
        N, M = kernel.shape
        
        if unbaised:
            # Exclude diagonal terms for unbiased estimation
            sum_kernel = kernel.sum(dim=(0, 1)) - torch.diagonal(
                kernel, dim1=0, dim2=1
            ).sum(dim=-1)
            mean_kernel = sum_kernel / (N * (N - 1))
        else:
            mean_kernel = kernel.mean(dim=(0, 1))
            
        return mean_kernel

    def _compute_kernel(
        self, 
        z0: torch.Tensor, 
        z1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RBF kernel matrix between two sets of samples.

        Parameters
        ----------
        z0 : torch.Tensor
            First set of samples
        z1 : torch.Tensor
            Second set of samples

        Returns
        -------
        torch.Tensor
            Kernel matrix
        """
        batch_size, z_size = z0.shape
        
        # Expand tensors for pairwise computation
        z0 = z0.unsqueeze(-2)
        z1 = z1.unsqueeze(-3)
        z0 = z0.expand(batch_size, batch_size, z_size)
        z1 = z1.expand(batch_size, batch_size, z_size)
        
        kernel = self._kernel_rbf(z0, z1)
        return kernel

    def _kernel_rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF (Radial Basis Function) kernel.

        Parameters
        ----------
        x : torch.Tensor
            First input tensor
        y : torch.Tensor
            Second input tensor

        Returns
        -------
        torch.Tensor
            RBF kernel values
        """
        z_size = x.shape[-1]
        sigma = 2 * 2 * z_size  # Bandwidth parameter
        kernel = torch.exp(-((x - y).pow(2).sum(dim=-1) / sigma))
        return kernel


class dipMixin:
    """
    Mixin class providing Disentangled Information Processing (DIP) functionality.
    
    This mixin implements the DIP loss function that encourages disentanglement
    by regularizing the covariance structure of the latent representations.
    """

    def _dip_loss(
        self, 
        q_m: torch.Tensor, 
        q_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DIP (Disentangled Information Processing) loss.

        This loss encourages disentanglement by:
        1. Making diagonal elements of covariance matrix close to 1
        2. Making off-diagonal elements close to 0

        Parameters
        ----------
        q_m : torch.Tensor
            Mean of latent distribution
        q_s : torch.Tensor
            Log variance of latent distribution

        Returns
        -------
        torch.Tensor
            DIP loss value
        """
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        
        # Penalize deviation of diagonal from 1
        dip_loss_d = torch.sum((cov_diag - 1) ** 2)
        
        # Penalize non-zero off-diagonal elements
        dip_loss_od = torch.sum(cov_off_diag ** 2)
        
        # Weighted combination (higher weight for diagonal regularization)
        dip_loss = 10 * dip_loss_d + 5 * dip_loss_od
        return dip_loss

    def _dip_cov_matrix(
        self, 
        q_m: torch.Tensor, 
        q_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute covariance matrix for DIP loss computation.

        Parameters
        ----------
        q_m : torch.Tensor
            Mean of latent distribution
        q_s : torch.Tensor
            Log variance of latent distribution

        Returns
        -------
        torch.Tensor
            Covariance matrix
        """
        # Covariance of means across batch
        cov_q_mean = torch.cov(q_m.T)
        
        # Expected variance (diagonal term)
        E_var = torch.mean(torch.diag(q_s.exp()), dim=0)
        
        # Total covariance = covariance of means + expected variance
        cov_matrix = cov_q_mean + E_var
        return cov_matrix


class envMixin:
    """
    Mixin class providing environment evaluation metrics.
    
    This mixin implements various clustering and correlation metrics for
    evaluating the quality of learned representations.
    """

    def _calc_score(self, latent: np.ndarray) -> Tuple[float, ...]:
        """
        Calculate comprehensive evaluation scores for latent representations.

        Parameters
        ----------
        latent : np.ndarray
            Latent representations to evaluate

        Returns
        -------
        Tuple[float, ...]
            Tuple of evaluation metrics (ARI, NMI, ASW, C_H, D_B, P_C)
        """
        n = latent.shape[1]
        labels = self._calc_label(latent)
        scores = self._metrics(latent, labels)
        return scores

    def _calc_label(self, latent: np.ndarray) -> np.ndarray:
        """
        Generate cluster labels using K-means clustering.

        Parameters
        ----------
        latent : np.ndarray
            Latent representations to cluster

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        labels = KMeans(latent.shape[1]).fit_predict(latent)
        return labels

    def _calc_corr(self, latent: np.ndarray) -> float:
        """
        Calculate average absolute correlation between latent dimensions.

        This metric measures the linear dependence between latent dimensions.
        Lower values indicate better disentanglement.

        Parameters
        ----------
        latent : np.ndarray
            Latent representations

        Returns
        -------
        float
            Average absolute correlation (excluding diagonal)
        """
        acorr = abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1

    def _metrics(
        self, 
        latent: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Compute comprehensive evaluation metrics.

        Parameters
        ----------
        latent : np.ndarray
            Latent representations
        labels : np.ndarray
            Predicted cluster labels

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            Evaluation metrics:
            - ARI: Adjusted Rand Index
            - NMI: Normalized Mutual Information
            - ASW: Average Silhouette Width
            - C_H: Calinski-Harabasz Index
            - D_B: Davies-Bouldin Index
            - P_C: Pearson Correlation (average absolute)
        """
        ARI = adjusted_mutual_info_score(self.labels[self.idx], labels)
        NMI = normalized_mutual_info_score(self.labels[self.idx], labels)
        ASW = silhouette_score(latent, labels)
        C_H = calinski_harabasz_score(latent, labels)
        D_B = davies_bouldin_score(latent, labels)
        P_C = self._calc_corr(latent)
        
        return ARI, NMI, ASW, C_H, D_B, P_C

