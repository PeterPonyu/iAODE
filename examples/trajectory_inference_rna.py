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
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

# Store raw counts BEFORE any transformation
if issparse(adata.X):
    adata.layers['counts'] = adata.X.copy()
else:
    adata.layers['counts'] = np.asarray(adata.X.copy())

# Apply normalization for downstream Scanpy analysis
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

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
    use_ode=True,
    i_dim=2,
    encoder_type='mlp',
    loss_mode='nb',
    batch_size=128
)

model.fit(epochs=50, patience=20, val_every=5)

metrics = model.get_resource_metrics()
print_success(f"Trained in {metrics['train_time']:.2f}s ({metrics['actual_epochs']} epochs)")

# ==================================================
# Extract Representations
# ==================================================

print_section("Extracting trajectory representations")

latent = model.get_latent()
iembed = model.get_iembed()
pseudotime = model.get_pseudotime()

adata.obsm['X_iaode'] = latent
adata.obsm['X_iembed'] = iembed
adata.obs['pseudotime'] = pseudotime

# Compute UMAP on latent space
n_neighbors = 30
sc.pp.neighbors(adata, use_rep='X_iaode', n_neighbors=n_neighbors)
sc.tl.umap(adata, min_dist=0.3)

print_success(f"Latent: {latent.shape}, I-embed: {iembed.shape}")
print_success(f"Pseudotime range: [{pseudotime.min():.3f}, {pseudotime.max():.3f}]")

# ==================================================
# Velocity Field Computation
# ==================================================

print_section("Computing velocity field")

E_grid, V_grid = model.get_vfres(
    adata,
    zs_key='X_iaode',
    E_key='X_umap',
    vf_key='X_vf_latent',
    dv_key='X_vf_umap',
    stream=True,
    density=1.5,
    smooth=0.5,
    n_neigh=n_neighbors,
    run_neigh=False,
)

velocity_umap = adata.obsm['X_vf_umap']
velocity_magnitude = np.linalg.norm(velocity_umap, axis=1)
adata.obs['velocity_magnitude'] = velocity_magnitude

print_success(f"Velocity field computed: {velocity_umap.shape}")

# Normalize for visualization
pseudotime_norm = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
adata.obs['pseudotime_norm'] = pseudotime_norm

# ==================================================
# Publication-Quality Multipanel Figure
# ==================================================

print_section("Generating publication-quality figure")

# Set global style matching LSE module
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Ubuntu', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Create comprehensive multipanel figure (3x3 grid)
fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, 
                       left=0.06, right=0.98, 
                       top=0.96, bottom=0.06,
                       hspace=0.38, wspace=0.35)

# Get UMAP coordinates and limits
umap_coords = adata.obsm['X_umap']
x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
padding = 0.05
x_range = x_max - x_min
y_range = y_max - y_min
xlim = [x_min - padding * x_range, x_max + padding * x_range]
ylim = [y_min - padding * y_range, y_max + padding * y_range]

def style_umap_ax(ax, xlabel=True, ylabel=True):
    """Apply consistent styling to UMAP axes"""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
    if ylabel:
        ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)

def add_colorbar(fig, ax, scatter, label, size="3%", pad=0.08):
    """Add consistent colorbar to axis"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label(label, fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9, width=1.0)
    cbar.outline.set_linewidth(1.2)
    return cbar

# Row 1: Main trajectory visualizations
# Panel A: Pseudotime
ax1 = fig.add_subplot(gs[0, 0])
scatter1 = ax1.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c=pseudotime_norm, cmap='viridis',
                       s=12, alpha=0.8, edgecolors='none', rasterized=True)
style_umap_ax(ax1)
ax1.set_title('A. Pseudotime Trajectory', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax1, scatter1, 'Pseudotime')

# Panel B: Cell Types
ax2 = fig.add_subplot(gs[0, 1])
if 'paul15_clusters' in adata.obs.columns:
    categories = adata.obs['paul15_clusters'].cat.categories
    n_cats = len(categories)
    if n_cats <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_cats]
    elif n_cats <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_cats]
    else:
        colors = plt.cm.gist_ncar(np.linspace(0, 1, n_cats))
    
    for i, cat in enumerate(categories):
        mask = adata.obs['paul15_clusters'] == cat
        label = str(cat)
        if len(label) > 15:
            label = label[:14] + '…'
        ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=[colors[i]], label=label, s=12, alpha=0.8, 
                   edgecolors='none', rasterized=True)
    
    # Position legend below the plot
    n_cols = min(n_cats, 4)  # Max 4 columns for readability
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=n_cols, frameon=True, fontsize=7, 
              markerscale=2.0, columnspacing=1.0,
              handletextpad=0.3, framealpha=0.95, 
              edgecolor='black', fancybox=False)
    ax2.set_title('B. Cell Type Annotations', fontsize=12, fontweight='bold', loc='left', pad=10)
else:
    scatter2 = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1],
                          c=iembed[:, 0], cmap='coolwarm',
                          s=12, alpha=0.8, edgecolors='none', rasterized=True)
    add_colorbar(fig, ax2, scatter2, 'I-embed 1')
    ax2.set_title('B. I-embed Dimension 1', fontsize=12, fontweight='bold', loc='left', pad=10)
style_umap_ax(ax2)

# Panel C: Velocity Magnitude
ax3 = fig.add_subplot(gs[0, 2])
scatter3 = ax3.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c=velocity_magnitude, cmap='plasma',
                       s=12, alpha=0.8, edgecolors='none', rasterized=True)
style_umap_ax(ax3)
ax3.set_title('C. Velocity Magnitude', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax3, scatter3, 'Magnitude')

# Row 2: Velocity field visualizations
# Panel D: Streamplot (spanning 2 columns)
ax4 = fig.add_subplot(gs[1, :2])
scatter4 = ax4.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c=pseudotime_norm, cmap='viridis',
                       s=10, alpha=0.6, edgecolors='none', 
                       rasterized=True, zorder=1)
stream = ax4.streamplot(E_grid[0], E_grid[1], V_grid[0], V_grid[1],
                        color='#E63946', density=1.8, linewidth=1.5,
                        arrowsize=1.8, arrowstyle='->', zorder=2)
style_umap_ax(ax4)
ax4.set_title('D. Velocity Field - Streamplot', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax4, scatter4, 'Pseudotime', size="1.5%", pad=0.05)

# Panel E: Quiver Plot
ax5 = fig.add_subplot(gs[1, 2])
scatter5 = ax5.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c=pseudotime_norm, cmap='viridis',
                       s=10, alpha=0.6, edgecolors='none', 
                       rasterized=True, zorder=1)
step = max(1, adata.n_obs // 150)
ax5.quiver(umap_coords[::step, 0], umap_coords[::step, 1],
          velocity_umap[::step, 0], velocity_umap[::step, 1],
          color='#E63946', alpha=0.8, width=0.004,
          headwidth=4.5, headlength=5.5, scale=None, 
          scale_units='xy', zorder=2)
style_umap_ax(ax5)
ax5.set_title('E. Velocity Field - Quiver', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax5, scatter5, 'Pseudotime')

# Row 3: Distribution analyses and interpretable space
# Panel F: Pseudotime Distribution
ax6 = fig.add_subplot(gs[2, 0])
counts, bins, patches = ax6.hist(pseudotime_norm, bins=40, 
                                 color='#0173B2', alpha=0.75, 
                                 edgecolor='black', linewidth=0.8)
ax6.axvline(pseudotime_norm.mean(), color='#E63946', 
           linestyle='--', linewidth=2.5, label=f'Mean: {pseudotime_norm.mean():.3f}')
ax6.axvline(np.median(pseudotime_norm), color='#029E73', 
           linestyle='--', linewidth=2.5, label=f'Median: {np.median(pseudotime_norm):.3f}')
ax6.set_xlabel('Normalized Pseudotime', fontsize=11, fontweight='bold')
ax6.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
ax6.set_title('F. Pseudotime Distribution', fontsize=12, fontweight='bold', loc='left', pad=10)
ax6.legend(fontsize=8, frameon=True, edgecolor='black', framealpha=0.95, loc='upper right')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_linewidth(1.2)
ax6.spines['bottom'].set_linewidth(1.2)
ax6.tick_params(width=1.2, labelsize=10)
ax6.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)

# Panel G: Velocity Magnitude Distribution
ax7 = fig.add_subplot(gs[2, 1])
counts, bins, patches = ax7.hist(velocity_magnitude, bins=40, 
                                 color='#DE8F05', alpha=0.75, 
                                 edgecolor='black', linewidth=0.8)
ax7.axvline(velocity_magnitude.mean(), color='#E63946', 
           linestyle='--', linewidth=2.5, label=f'Mean: {velocity_magnitude.mean():.3f}')
ax7.axvline(np.median(velocity_magnitude), color='#029E73', 
           linestyle='--', linewidth=2.5, label=f'Median: {np.median(velocity_magnitude):.3f}')
ax7.set_xlabel('Velocity Magnitude', fontsize=11, fontweight='bold')
ax7.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
ax7.set_title('G. Velocity Distribution', fontsize=12, fontweight='bold', loc='left', pad=10)
ax7.legend(fontsize=8, frameon=True, edgecolor='black', framealpha=0.95, loc='upper right')
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.spines['left'].set_linewidth(1.2)
ax7.spines['bottom'].set_linewidth(1.2)
ax7.tick_params(width=1.2, labelsize=10)
ax7.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)

# Panel H: I-embed 2D Space with Pseudotime
ax8 = fig.add_subplot(gs[2, 2])
scatter8 = ax8.scatter(iembed[:, 0], iembed[:, 1],
                       c=pseudotime_norm, cmap='viridis',
                       s=12, alpha=0.8, edgecolors='none', rasterized=True)
ax8.set_xlabel('I-embed Dimension 1', fontsize=11, fontweight='bold')
ax8.set_ylabel('I-embed Dimension 2', fontsize=11, fontweight='bold')
ax8.set_title('H. Interpretable Bottleneck', fontsize=12, fontweight='bold', loc='left', pad=10)
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
ax8.spines['left'].set_linewidth(1.2)
ax8.spines['bottom'].set_linewidth(1.2)
ax8.tick_params(width=1.2, labelsize=10)
ax8.grid(alpha=0.3, linestyle='--', linewidth=0.6)
add_colorbar(fig, ax8, scatter8, 'Pseudotime')

# Save figure
plt.savefig(OUTPUT_DIR / 'trajectory_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'trajectory_analysis.pdf', dpi=300, bbox_inches='tight')
plt.close()

print_success(f"Saved: {OUTPUT_DIR}/trajectory_analysis.png")
print_success(f"Saved: {OUTPUT_DIR}/trajectory_analysis.pdf")

# ==================================================
# Summary Statistics
# ==================================================

print_header("Analysis Complete")
print_info("Neural ODE Trajectory Inference Summary")
print()
print(f"  Dataset:")
print(f"    Cells: {adata.n_obs:,}")
print(f"    Genes: {adata.n_vars:,}")
print()
print(f"  Training:")
print(f"    Time: {metrics['train_time']:.2f}s ({metrics['train_time']/metrics['actual_epochs']:.2f}s/epoch)")
print(f"    Epochs: {metrics['actual_epochs']}")
print(f"    GPU Memory: {metrics['peak_memory_gb']:.3f} GB")
print()
print(f"  Pseudotime Statistics:")
print(f"    Range: [{pseudotime.min():.4f}, {pseudotime.max():.4f}]")
print(f"    Mean:  {pseudotime.mean():.4f} ± {pseudotime.std():.4f}")
print(f"    Median: {np.median(pseudotime):.4f}")
print()

# Compute additional velocity statistics
velocity_latent = adata.obsm['X_vf_latent']
velocity_mag_latent = np.linalg.norm(velocity_latent, axis=1)

print(f"  Velocity Statistics:")
print(f"    Latent space:")
print(f"      Mean magnitude: {velocity_mag_latent.mean():.5f} ± {velocity_mag_latent.std():.5f}")
print(f"      Max magnitude:  {velocity_mag_latent.max():.5f}")
print(f"    UMAP space:")
print(f"      Mean magnitude: {velocity_magnitude.mean():.5f} ± {velocity_magnitude.std():.5f}")
print(f"      Max magnitude:  {velocity_magnitude.max():.5f}")
print()

# Correlation between pseudotime and velocity
from scipy.stats import pearsonr, spearmanr
pearson_corr, pearson_pval = pearsonr(pseudotime_norm, velocity_magnitude)
spearman_corr, spearman_pval = spearmanr(pseudotime_norm, velocity_magnitude)

print(f"  Pseudotime-Velocity Correlation:")
print(f"    Pearson:  r = {pearson_corr:.4f} (p = {pearson_pval:.2e})")
print(f"    Spearman: ρ = {spearman_corr:.4f} (p = {spearman_pval:.2e})")
print()

print_info("Output files:")
print(f"  • {OUTPUT_DIR}/trajectory_analysis.png")
print(f"  • {OUTPUT_DIR}/trajectory_analysis.pdf")
print()