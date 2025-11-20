"""
Trajectory Inference Example - Neural ODE

This example demonstrates trajectory inference using Neural ODE for
time-series single-cell data with velocity field visualization.

Dataset: paul15 (hematopoietic differentiation trajectory)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _example_utils import (
    check_iaode_installed, setup_output_dir,
    print_header, print_section, print_success, print_info
)

if not check_iaode_installed():
    sys.exit(1)

import iaode
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = setup_output_dir("trajectory_inference")
print_info(f"Outputs saved to: {OUTPUT_DIR}")

# ==================================================
# Load Data
# ==================================================

print_header("Trajectory Inference with Neural ODE")
print_section("Loading paul15 dataset")

adata = sc.datasets.paul15()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

print_success(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

# ==================================================
# Train with Neural ODE
# ==================================================

print_section("Training iAODE with Neural ODE")
print_info("Configuration:")
print("  use_ode=True  → Enable Neural ODE for trajectory")
print("  i_dim=2       → Intermediate ODE state dimension")
print("  latent_dim=10 → Final latent dimension")
print()

model = iaode.agent(
    adata, layer='counts',
    latent_dim=10, hidden_dim=128,
    use_ode=True,      # Enable Neural ODE
    i_dim=2,           # ODE intermediate dimension
    encoder_type='mlp',
    loss_mode='nb',
    batch_size=128
)

model.fit(epochs=400, patience=20, val_every=5)

metrics = model.get_resource_metrics()
print_success(f"Trained in {metrics['train_time']:.2f}s")

# ==================================================
# Extract Representations
# ==================================================

print_section("Extracting trajectory representations")

latent = model.get_latent()
iembed = model.get_iembed()  # ODE intermediate state
pseudotime = model.get_pseudotime()  # ODE time parameter
velocity = model.get_velocity()  # ODE gradients in latent space

adata.obsm['X_iaode'] = latent
adata.obsm['X_iembed'] = iembed
adata.obsm['velocity'] = velocity
adata.obs['pseudotime'] = pseudotime

# Compute UMAP on latent space (not iembed)
sc.pp.neighbors(adata, use_rep='X_iaode', n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3)

print_success(f"Latent: {latent.shape}, I-embed: {iembed.shape}")
print_success(f"Pseudotime: {pseudotime.shape}, Velocity: {velocity.shape}")

# ==================================================
# Pseudotime Analysis
# ==================================================

print_section("Analyzing pseudotime trajectory")

# Normalize pseudotime to [0, 1]
pseudotime_norm = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
adata.obs['pseudotime_norm'] = pseudotime_norm

print(f"  Pseudotime range: [{pseudotime.min():.3f}, {pseudotime.max():.3f}]")
print(f"  Pseudotime mean: {pseudotime.mean():.3f} ± {pseudotime.std():.3f}")

# ==================================================
# Visualizations
# ==================================================

print_section("Generating visualizations")
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 10})

# Plot 1: UMAP with pseudotime
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc.pl.umap(adata, color='pseudotime_norm', cmap='viridis',
           title='Pseudotime Trajectory (ODE Time)', ax=axes[0], show=False, frameon=True)

if 'paul15_clusters' in adata.obs.columns:
    sc.pl.umap(adata, color='paul15_clusters',
               title='Cell Types', ax=axes[1], show=False, frameon=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trajectory_umap.png', dpi=300, bbox_inches='tight')
plt.close()
print_success(f"Saved: {OUTPUT_DIR}/trajectory_umap.png")

# Plot 2: Velocity field on UMAP
print_section("Computing velocity field visualization")

# Get velocity from model (ODE gradients in latent space)
n_cells = adata.n_obs
umap_coords = adata.obsm['X_umap']

# Project velocity to UMAP space using linear approximation
# Compute UMAP basis vectors by comparing neighbors
from sklearn.decomposition import PCA

# Use PCA to reduce velocity from latent_dim to 2D for UMAP visualization
pca = PCA(n_components=2)
velocity_2d = pca.fit_transform(velocity)

# Scale for visualization
velocity_umap = velocity_2d * 0.05

# Create velocity plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Quiver plot
ax = axes[0]
scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                     c=pseudotime_norm, cmap='viridis', s=30, alpha=0.6)
# Subsample for better visualization
step = max(1, n_cells // 300)  # Show ~300 arrows
quiver = ax.quiver(umap_coords[::step, 0], umap_coords[::step, 1],
                   velocity_umap[::step, 0], velocity_umap[::step, 1],
                   color='red', alpha=0.6, width=0.006, scale=15, 
                   headwidth=4, headlength=5, headaxislength=4.5)
ax.set_title('Velocity Field (Quiver) - ODE Gradients', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
plt.colorbar(scatter, ax=ax, label='Pseudotime')

# Streamplot with ODE-based velocity
ax = axes[1]
x = umap_coords[:, 0]
y = umap_coords[:, 1]

# Create grid for streamplot using actual ODE velocity
grid_x, grid_y = np.mgrid[x.min():x.max():40j, y.min():y.max():40j]
grid_u = griddata((x, y), velocity_umap[:, 0], (grid_x, grid_y), 
                  method='linear', fill_value=0)
grid_v = griddata((x, y), velocity_umap[:, 1], (grid_x, grid_y), 
                  method='linear', fill_value=0)

scatter = ax.scatter(x, y, c=pseudotime_norm, cmap='viridis', s=30, alpha=0.6, zorder=2)
ax.streamplot(grid_x[:, 0], grid_y[0, :], grid_u, grid_v,
              color='red', density=1.2, linewidth=1.2, arrowsize=1.5, zorder=1)
ax.set_title('Velocity Field (Streamplot) - ODE Gradients', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
plt.colorbar(scatter, ax=ax, label='Pseudotime')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'velocity_field.png', dpi=300, bbox_inches='tight')
plt.close()
print_success(f"Saved: {OUTPUT_DIR}/velocity_field.png")

print_header("Complete")
print_info("Neural ODE captured trajectory dynamics with pseudotime and velocity")
print_info(f"Pseudotime range: [{pseudotime.min():.3f}, {pseudotime.max():.3f}]")
print_info(f"Velocity magnitude: mean={np.linalg.norm(velocity, axis=1).mean():.3f}")

