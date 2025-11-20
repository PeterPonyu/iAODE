# mixin.py

import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from typing import Optional


class scviMixin:
    def _normal_kl(self, mu1, lv1, mu2, lv2):
        """
        KL divergence between two Gaussian distributions.

        Args:
            mu1, mu2: means of the two distributions
            lv1, lv2: log-variances of the two distributions

        Returns:
            KL divergence tensor
        """
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.0
        lstd2 = lv2 / 2.0
        kl = lstd2 - lstd1 + (v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2) - 0.5
        return kl

    def _log_nb(self, x, mu, theta, eps=1e-8):
        """
        Log-likelihood under a negative binomial distribution.

        Args:
            x: observed counts
            mu: mean of the distribution
            theta: dispersion parameter
            eps: numerical stability constant

        Returns:
            Log-likelihood tensor
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

    def _log_zinb(self, x, mu, theta, pi, eps=1e-8):
        """
        Log-likelihood under a zero-inflated negative binomial distribution.

        Args:
            x: observed counts
            mu: mean of the NB component
            theta: dispersion parameter of the NB component
            pi: logits for the zero-inflation component
            eps: numerical stability constant

        Returns:
            Log-likelihood tensor
        """
        softplus_pi = F.softplus(-pi)
        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

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
    Mixin providing Neural ODE utilities.
    """

    @staticmethod
    def get_step_size(step_size, t0, t1, n_points):
        """
        Build step size options for the ODE solver.

        Args:
            step_size: step size; if None, use solver defaults;
                       if "auto", compute uniform step size from (t0, t1, n_points)
            t0: initial time
            t1: final time
            n_points: number of time points

        Returns:
            dict of ODE solver options
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
        Solve an ODE using torchdiffeq.

        Args:
            ode_func: ODE function module
            z0: initial state
            t: time points
            method: ODE solver method
            step_size: step size for fixed-step solvers

        Returns:
            Tensor of ODE solutions over time
        """
        options = self.get_step_size(step_size, t[0], t[-1], len(t))

        # Force CPU because some ODE solvers are unstable on GPU
        cpu_z0 = z0.to("cpu")
        cpu_t = t.to("cpu")

        pred_z = odeint(ode_func, cpu_z0, cpu_t, method=method, options=options)

        # Move result back to original device
        pred_z = pred_z.to(z0.device)

        return pred_z


class betatcMixin:
    def _betatc_compute_gaussian_log_density(self, samples, mean, log_var):
        r"""
        Compute log-density of a diagonal Gaussian.

        Args:
            samples: samples \(z\)
            mean: mean \(\mu\)
            log_var: log-variance \(\log \sigma^2\)

        Returns:
            Log-density tensor
        """
        import math

        pi = torch.tensor(math.pi, device=samples.device)
        normalization = torch.log(2 * pi)
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

    def _betatc_compute_total_correlation(self, z_sampled, z_mean, z_logvar):
        """
        Estimate total correlation term for β-TCVAE.

        Args:
            z_sampled: sampled latent codes, shape [B, D]
            z_mean: means of approximate posterior, shape [B, D]
            z_logvar: log-variances of approximate posterior, shape [B, D]

        Returns:
            Scalar total correlation estimate
        """
        batch_size = z_sampled.size(0)

        # log_qz_prob: [B, B, D]
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(dim=1),  # [B, 1, D]
            z_mean.unsqueeze(dim=0),     # [1, B, D]
            z_logvar.unsqueeze(dim=0),   # [1, B, D]
        )

        # Clamp for numerical stability
        log_qz_prob = torch.clamp(log_qz_prob, min=-1000, max=1000)

        # log q(z) = logsumexp_b sum_d log q(z_b | x_b) - log B
        log_qz = torch.logsumexp(log_qz_prob.sum(dim=2), dim=1) - math.log(batch_size)

        # log ∏_j q(z_j) = ∑_j [logsumexp_b log q(z_j | x_b) - log B]
        log_qz_product = (
            torch.logsumexp(log_qz_prob, dim=1) - math.log(batch_size)
        ).sum(dim=1)

        tc = (log_qz - log_qz_product).mean()
        return tc


class infoMixin:
    def _compute_mmd(self, z_posterior_samples, z_prior_samples):
        """
        Compute MMD between posterior and prior samples.
        """
        mean_pz_pz = self._compute_unbiased_mean(
            self._compute_kernel(z_prior_samples, z_prior_samples), unbaised=True
        )
        mean_pz_qz = self._compute_unbiased_mean(
            self._compute_kernel(z_prior_samples, z_posterior_samples), unbaised=False
        )
        mean_qz_qz = self._compute_unbiased_mean(
            self._compute_kernel(z_posterior_samples, z_posterior_samples),
            unbaised=True,
        )
        mmd = mean_pz_pz - 2 * mean_pz_qz + mean_qz_qz
        return mmd

    def _compute_unbiased_mean(self, kernel, unbaised):
        """
        Compute (optionally) unbiased mean of kernel entries.
        """
        N, M = kernel.shape
        if unbaised:
            sum_kernel = kernel.sum(dim=(0, 1)) - torch.diagonal(
                kernel, dim1=0, dim2=1
            ).sum(dim=-1)
            mean_kernel = sum_kernel / (N * (N - 1))
        else:
            mean_kernel = kernel.mean(dim=(0, 1))
        return mean_kernel

    def _compute_kernel(self, z0, z1):
        """
        Build RBF kernel matrix between two batches of latent codes.
        """
        batch_size, z_size = z0.shape
        z0 = z0.unsqueeze(-2)
        z1 = z1.unsqueeze(-3)
        z0 = z0.expand(batch_size, batch_size, z_size)
        z1 = z1.expand(batch_size, batch_size, z_size)
        kernel = self._kernel_rbf(z0, z1)
        return kernel

    def _kernel_rbf(self, x, y):
        """
        Radial basis function (Gaussian) kernel.
        """
        z_size = x.shape[-1]
        sigma = 2 * 2 * z_size
        kernel = torch.exp(-((x - y).pow(2).sum(dim=-1) / sigma))
        return kernel


class dipMixin:
    def _dip_loss(self, q_m, q_s):
        """
        DIP-VAE loss based on covariance of latent variables.

        Args:
            q_m: latent means, shape [B, D]
            q_s: latent log-variances, shape [B, D]

        Returns:
            Scalar DIP loss
        """
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        dip_loss_d = torch.sum((cov_diag - 1) ** 2)
        dip_loss_od = torch.sum(cov_off_diag**2)
        dip_loss = 10 * dip_loss_d + 5 * dip_loss_od
        return dip_loss

    def _dip_cov_matrix(self, q_m, q_s):
        """
        Covariance matrix used in DIP-VAE.

        Args:
            q_m: latent means, shape [B, D]
            q_s: latent log-variances, shape [B, D]

        Returns:
            Covariance matrix of shape [D, D]
        """
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.diag(q_s.exp()), dim=0)
        cov_matrix = cov_q_mean + E_var
        return cov_matrix


class envMixin:
    def _calc_score(self, latent):
        """
        Compute clustering scores using stored labels (backward compatible).
        """
        labels = self._calc_label(latent)
        scores = self._metrics(latent, labels, self.labels[self.idx])
        return scores

    def _calc_score_with_labels(self, latent, true_labels):
        """
        Compute clustering scores using provided ground truth labels.
        """
        predicted_labels = self._calc_label(latent)
        scores = self._metrics(latent, predicted_labels, true_labels)
        return scores

    def _calc_label(self, latent):
        """
        Cluster latent space with k-means.

        Uses the latent dimensionality as the number of clusters.
        """
        labels = KMeans(latent.shape[1]).fit_predict(latent)
        return labels

    def _calc_corr(self, latent):
        """
        Mean absolute correlation of latent dimensions (excluding self-correlation).
        """
        acorr = abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1

    def _metrics(self, latent, predicted_labels, true_labels):
        """
        Compute clustering and structure metrics.

        Returns:
            tuple: (ARI, NMI, silhouette, Calinski–Harabasz, Davies–Bouldin, mean correlation)
        """
        ARI = adjusted_mutual_info_score(true_labels, predicted_labels)
        NMI = normalized_mutual_info_score(true_labels, predicted_labels)
        ASW = silhouette_score(latent, predicted_labels)
        CAL = calinski_harabasz_score(latent, predicted_labels)
        DAV = davies_bouldin_score(latent, predicted_labels)
        COR = self._calc_corr(latent)
        return ARI, NMI, ASW, CAL, DAV, COR