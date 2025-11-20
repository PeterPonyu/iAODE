"""
Basic Usage Example - scRNA-seq Dimensionality Reduction

This example demonstrates basic iAODE model training for scRNA-seq data
with standard preprocessing and UMAP visualization.

Dataset: paul15 (2730 cells, hematopoietic differentiation)
"""

import sys
from pathlib import Path

# Check iaode installation
sys.path.insert(0, str(Path(__file__).parent))
from _example_utils import (
    check_iaode_installed, setup_output_dir,
    print_header, print_section, print_success, print_info
)

if not check_iaode_installed():
    sys.exit(1)

import iaode
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = setup_output_dir("basic_usage")
print_info(f"Outputs saved to: {OUTPUT_DIR}")

# ==================================================
# Load and Preprocess Data
# ==================================================

print_header("Basic iAODE Usage - scRNA-seq")
print_section("Loading paul15 dataset")

adata = sc.datasets.paul15()
print(f"  Original: {adata.n_obs} cells × {adata.n_vars} genes")

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

print_success(f"Preprocessed: {adata.n_obs} cells × {adata.n_vars} genes")

# ==================================================
# Train Model
# ==================================================

print_section("Training iAODE model")
print_info("Hyperparameters:")
print("  latent_dim=10      → Latent dimensionality")
print("  hidden_dim=128     → Hidden layer size") 
print("  encoder_type='mlp' → Options: 'mlp', 'residual_mlp', 'transformer', 'linear'")
print("  loss_mode='nb'     → Options: 'mse', 'nb', 'zinb'")
print()

model = iaode.agent(
    adata, layer='counts', latent_dim=10, hidden_dim=128,
    encoder_type='mlp', loss_mode='nb', batch_size=128
)

model.fit(epochs=100, patience=20, val_every=5)

metrics = model.get_resource_metrics()
print_success(f"Trained in {metrics['train_time']:.2f}s ({metrics['actual_epochs']} epochs)")

# ==================================================
# Visualize
# ==================================================

print_section("Generating UMAP visualizations")

latent = model.get_latent()
adata.obsm['X_iaode'] = latent

sc.pp.neighbors(adata, use_rep='X_iaode', n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3)

# Set visualization style
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 10})

# Plot cell types
if 'paul15_clusters' in adata.obs.columns:
    sc.settings.figdir = OUTPUT_DIR
    sc.pl.umap(adata, color='paul15_clusters', title='iAODE - Cell Types',
               frameon=True, save='_celltypes.png', show=False)
    print_success(f"Saved: {OUTPUT_DIR}/umap_celltypes.png")

# Plot latent dimensions
adata.obs['Latent_Dim1'] = latent[:, 0]
adata.obs['Latent_Dim2'] = latent[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata, color='Latent_Dim1', cmap='viridis', title='Latent Dim 1',
           ax=axes[0], show=False, frameon=True)
sc.pl.umap(adata, color='Latent_Dim2', cmap='plasma', title='Latent Dim 2',
           ax=axes[1], show=False, frameon=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'latent_dimensions.png', dpi=300, bbox_inches='tight')
plt.close()
print_success(f"Saved: {OUTPUT_DIR}/latent_dimensions.png")

print_header("Complete")
print_info("Next: Try trajectory_inference.py for Neural ODE analysis")
