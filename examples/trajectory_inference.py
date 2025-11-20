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
import numpy as np
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
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
raw_X = adata.X
try:
    adata.layers['counts'] = raw_X.copy()  # type: ignore[attr-defined]
except Exception:
    adata.layers['counts'] = np.asarray(raw_X)

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
adata.obs['pseudotime'] = pseudotime

# Compute UMAP on latent space
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

# ==================================================
# Plot 2: Velocity Field using get_vfres()
# ==================================================

print_section("Computing velocity field visualization")
print_info("Using model.get_vfres() for proper ODE-based vector field")

# Use the agent's built-in vector field computation
try:
    E_grid, V_grid = model.get_vfres(
        adata,
        zs_key='X_iaode',     # Latent space key
        E_key='X_umap',        # UMAP embedding key
        vf_key='X_velocity',   # Store velocity here
        stream=True,           # Return streamplot format
        density=1.5,           # Grid density
        smooth=0.5             # Smoothing parameter
    )
    
    # Create streamplot visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with pseudotime coloring
    scatter = ax.scatter(adata.obsm['X_umap'][:, 0], 
                        adata.obsm['X_umap'][:, 1],
                        c=pseudotime_norm, 
                        cmap='viridis', 
                        s=30, 
                        alpha=0.6,
                        zorder=2)
    
    # Streamplot overlay
    ax.streamplot(E_grid[0], E_grid[1], V_grid[0], V_grid[1],
                 color='red', density=1.2, linewidth=1.2, 
                 arrowsize=1.5, zorder=1)
    
    ax.set_title('Velocity Field (Neural ODE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Pseudotime')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'velocity_field_ode.png', dpi=300, bbox_inches='tight')
    plt.close()
    print_success(f"Saved: {OUTPUT_DIR}/velocity_field_ode.png")
    
except Exception as e:
    print_info(f"Note: get_vfres() requires use_ode=True and proper setup")
    print_info(f"Error: {e}")
    print_info("Falling back to basic velocity visualization")
    
    # Fallback: Basic quiver plot
    fig, ax = plt.subplots(figsize=(10, 8))
    umap_coords = adata.obsm['X_umap']
    
    # Project velocity to 2D using simple linear approximation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    velocity_2d = pca.fit_transform(velocity) * 0.05
    
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=pseudotime_norm, cmap='viridis', s=30, alpha=0.6)
    
    # Subsample for visualization
    step = max(1, adata.n_obs // 300)
    ax.quiver(umap_coords[::step, 0], umap_coords[::step, 1],
             velocity_2d[::step, 0], velocity_2d[::step, 1],
             color='red', alpha=0.6, width=0.006, scale=15)
    
    ax.set_title('Velocity Field (Fallback Projection)', fontsize=14, fontweight='bold')
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