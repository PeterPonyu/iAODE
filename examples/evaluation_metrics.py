"""
Evaluation Metrics Example

Computes dimensionality reduction quality and clustering metrics for an iAODE latent space.
Uses small synthetic data for demonstration.
"""
import numpy as np
import scanpy as sc
import iaode
from iaode.DRE import DimensionalityReductionEvaluator


def synthetic_data(n_cells=1200, n_peaks=6000, seed=1):
    rng = np.random.default_rng(seed)
    density = 0.003
    X = rng.binomial(2, density, size=(n_cells, n_peaks)).astype(np.float32)
    adata = sc.AnnData(X)
    raw_X = adata.X
    try:
        adata.layers['counts'] = raw_X.copy()  # type: ignore[attr-defined]
    except Exception:
        adata.layers['counts'] = np.asarray(raw_X)
    return adata


def main():
    adata = synthetic_data()
    print(f'Dataset: {adata.n_obs} cells × {adata.n_vars} peaks')

    model = iaode.agent(
        adata,
        layer='counts',
        latent_dim=20,
        hidden_dim=256,
        i_dim=10,
        loss_mode='zinb',
        use_ode=False,
        batch_size=128,
    )
    model.fit(epochs=250, patience=25, val_every=10)

    latent = model.get_latent()
    adata.obsm['X_iaode'] = latent

    # Dimensionality reduction metrics
    dre = DimensionalityReductionEvaluator(verbose=True)
    X_high = np.asarray(adata.X)
    metrics = dre.comprehensive_evaluation(X_high, latent, k=10)

    print('\nDimensionality Reduction Metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    print('\nValidation Metrics (from training):')
    print('  These are computed during model.fit() with val_every parameter')
   
if __name__ == '__main__':
    main()