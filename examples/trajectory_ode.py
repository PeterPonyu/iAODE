"""
Neural ODE Trajectory Example (scATAC or generic sparse single-cell counts)

Demonstrates enabling latent Neural ODE dynamics in iAODE and extracting:
- Pseudotime
- Latent velocity (from ODE dynamics)
- Interpretable bottleneck factors

Uses a small synthetic sparse matrix if no data provided.
"""
import argparse
import numpy as np
import scanpy as sc
import iaode


def synthetic_data(n_cells=1500, n_peaks=8000, seed=0):
    """Generate synthetic sparse accessibility data"""
    rng = np.random.default_rng(seed)
    # Simulate sparse accessibility with bursty structure
    density = 0.002
    X = rng.binomial(2, density, size=(n_cells, n_peaks)).astype(np.float32)
    adata = sc.AnnData(X)
    raw_X = adata.X
    try:
        adata.layers['counts'] = raw_X.copy()  # type: ignore[attr-defined]
    except Exception:
        adata.layers['counts'] = np.asarray(raw_X)
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent-dim', type=int, default=24)
    ap.add_argument('--hidden-dim', type=int, default=384)
    ap.add_argument('--i-dim', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    adata = synthetic_data(seed=args.seed)
    print(f'Synthetic dataset: {adata.n_obs} cells × {adata.n_vars} peaks')

    print('\n[Training Neural ODE model]')
    model = iaode.agent(
        adata,
        layer='counts',
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        i_dim=args.i_dim,
        use_ode=True,  # ✅ Enable Neural ODE for trajectory
        loss_mode='zinb',
        batch_size=128,
    )
    model.fit(epochs=args.epochs, patience=30, val_every=10)

    print('\n[Extracting representations]')
    # ✅ Use all built-in methods from agent.py
    latent = model.get_latent()        # Latent representation
    pseudo = model.get_pseudotime()    # ODE time parameter
    iembed = model.get_iembed()        # Interpretable bottleneck factors
    velocity = model.get_velocity()    # ✅ ODE-based velocity (NOT manual calculation)

    # Store in AnnData
    adata.obsm['X_iaode'] = latent
    adata.obsm['X_iembed'] = iembed
    adata.obs['pseudotime'] = pseudo
    adata.obsm['velocity'] = velocity

    # Save results
    adata.write_h5ad('trajectory_ode_output.h5ad')
    
    # Summary
    print('\n[Results]')
    print(f'  Latent shape: {latent.shape}')
    print(f'  Pseudotime range: [{pseudo.min():.3f}, {pseudo.max():.3f}]')
    print(f'  Velocity magnitude (mean): {np.linalg.norm(velocity, axis=1).mean():.4f}')
    print(f'\nSaved: trajectory_ode_output.h5ad')

if __name__ == '__main__':
    main()