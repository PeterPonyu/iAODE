"""
scATAC Quickstart: TF-IDF + Highly Variable Peaks + iAODE Training

Minimal end-to-end workflow for scATAC-seq peak count data using iAODE.
Provide a 10X filtered peak matrix (H5) and a matching gene annotation (GTF).

Steps:
1. Load peak count matrix (10X .h5)
2. (Optional) Peak-to-gene annotation via iaode.annotation_pipeline
3. TF-IDF normalization (utils.tfidf_normalization)
4. Highly variable peak selection (utils.select_highly_variable_peaks)
5. Train iAODE (NB/ZINB reconstruction + optional Neural ODE)
6. Extract latent, interpretable factors, pseudotime, velocity

Run:
    python scATAC_quickstart.py --h5 filtered_peak_bc_matrix.h5 --gtf gencode.vM25.annotation.gtf \
        --use-ode --loss-mode zinb --latent-dim 32 --hidden-dim 512 --i-dim 16

For a fast dry run (small subset): add --subsample 5000
"""
import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
import iaode
from iaode.utils import tfidf_normalization, select_highly_variable_peaks
from iaode.annotation import load_10x_h5_data  # ✅ Use built-in loader


def parse_args():
    p = argparse.ArgumentParser(description='iAODE scATAC quickstart')
    p.add_argument('--h5', required=True, help='10X filtered peak matrix .h5 file')
    p.add_argument('--gtf', required=False, help='Optional GTF for annotation')
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--hidden-dim', type=int, default=512)
    p.add_argument('--i-dim', type=int, default=16)
    p.add_argument('--loss-mode', choices=['nb','zinb','mse'], default='zinb')
    p.add_argument('--use-ode', action='store_true')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--val-every', type=int, default=10)
    p.add_argument('--subsample', type=int, default=None, help='Subsample cells for a quick test run')
    p.add_argument('--n-top-peaks', type=int, default=20000)
    return p.parse_args()


def main():
    args = parse_args()
    h5_file = Path(args.h5)
    if not h5_file.exists():
        raise FileNotFoundError(f'H5 file not found: {h5_file}')

    print('\n[1] Load peak counts')
    # ✅ Use built-in loader from annotation.py instead of custom h5py code
    adata = load_10x_h5_data(str(h5_file))
    raw_X = adata.X
    try:
        adata.layers['counts'] = raw_X.copy()  # type: ignore[attr-defined]
    except Exception:
        adata.layers['counts'] = np.asarray(raw_X)
    print(f'  Cells: {adata.n_obs:,}  Peaks: {adata.n_vars:,}')

    if args.subsample and args.subsample < adata.n_obs:
        idx = np.random.choice(adata.n_obs, args.subsample, replace=False)
        adata = adata[idx].copy()
        print(f'  Subsampled to {adata.n_obs} cells')

    if args.gtf:
        # Optional: enrich with annotation using existing pipeline
        print('\n[2] Peak annotation (optional)')
        adata = iaode.annotation_pipeline(
            h5_file=str(h5_file),
            gtf_file=str(args.gtf),
            apply_tfidf=False,  # Will run manually below
            select_hvp=False,
            promoter_upstream=2000,
            promoter_downstream=500,
        )
        raw_X2 = adata.X
        try:
            adata.layers['counts'] = raw_X2.copy()  # type: ignore[attr-defined]
        except Exception:
            adata.layers['counts'] = np.asarray(raw_X2)

    print('\n[3] TF-IDF normalization')
    tfidf_normalization(adata, scale_factor=1e4, log_tf=False, log_idf=True, inplace=True)

    print('\n[4] Highly variable peak selection')
    select_highly_variable_peaks(
        adata,
        n_top_peaks=args.n_top_peaks,
        method='signac',
        inplace=True,
    )
    if 'highly_variable' in adata.var.columns:
        hv_mask = adata.var['highly_variable'].values
    else:
        hv_mask = np.ones(adata.n_vars, dtype=bool)
    adata = adata[:, hv_mask].copy()
    print(f'  Retained HVPs: {adata.n_vars:,}')

    print('\n[5] Train iAODE model')
    model = iaode.agent(
        adata,
        layer='counts',
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        i_dim=args.i_dim,
        loss_mode=args.loss_mode,
        use_ode=args.use_ode,
        batch_size=256,
    )
    model.fit(epochs=args.epochs, patience=args.patience, val_every=args.val_every)

    metrics = model.get_resource_metrics()
    print(f"  Training time: {metrics['train_time']:.1f}s  Epochs: {metrics['actual_epochs']}")

    print('\n[6] Extract embeddings')
    latent = model.get_latent()
    iembed = model.get_iembed()
    adata.obsm['X_iaode'] = latent
    adata.obsm['X_iembed'] = iembed

    if args.use_ode:
        print('  Extract pseudotime & velocity')
        pseudo = model.get_pseudotime()
        velocity = model.get_velocity()  # ✅ Use built-in ODE-based velocity
        adata.obs['pseudotime'] = pseudo
        adata.obsm['velocity'] = velocity

    out_file = Path('iaode_scATAC_processed.h5ad')
    adata.write_h5ad(out_file)
    print(f'\nSaved AnnData with embeddings: {out_file}')

if __name__ == '__main__':
    main()