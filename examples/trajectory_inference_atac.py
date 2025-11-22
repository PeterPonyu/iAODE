"""
Trajectory inference with Neural ODE (scATAC-seq)

Infer trajectories from chromatin accessibility using iAODE + Neural ODE
with velocity field visualization.

Dataset: 10X Mouse Brain 5k scATAC-seq (HVP subset)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _example_utils import (
    check_iaode_installed, setup_output_dir,
    print_header, print_section, print_success, print_info, print_warning
)

if not check_iaode_installed():
    sys.exit(1)

import iaode
import numpy as np
import scanpy as sc  # type: ignore
from scipy.sparse import issparse  # type: ignore
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = setup_output_dir("trajectory_inference_atac")
print_info(f"Saving outputs to: {OUTPUT_DIR}")
print()

# ==================================================
# Load and Annotate scATAC Data
# ==================================================

print_header("Trajectory inference with Neural ODE (scATAC-seq)")

print_section("Loading and annotating scATAC-seq data")
print_info("Dataset: 10X Mouse Brain 5k scATAC-seq")

# Download and annotate data
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()

adata = iaode.annotation_pipeline(
    h5_file=str(h5_file),
    gtf_file=str(gtf_file),
    promoter_upstream=2000,
    promoter_downstream=500,
    apply_tfidf=True,
    select_hvp=True,
    n_top_peaks=20000
)

print_success(f"Annotated: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")

# ==================================================
# Subset to HVPs Only
# ==================================================

print_section("Subsetting to highly variable peaks (HVPs)")

if 'highly_variable' in adata.var.columns:
    n_total_peaks = adata.n_vars
    n_hvp = adata.var['highly_variable'].sum()
    
    adata = adata[:, adata.var['highly_variable']].copy()
    
    print_success(f"HVP subset: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
    print_info(f"  Retained {n_hvp:,} / {n_total_peaks:,} peaks ({n_hvp/n_total_peaks*100:.1f}%)")
else:
    print_warning("No HVP flag found; using all peaks")

print()

# Ensure counts layer exists (TF-IDF normalized)
if 'counts' not in adata.layers:
    if issparse(adata.X):
        adata.layers['counts'] = adata.X.copy()
    else:
        adata.layers['counts'] = np.asarray(adata.X.copy())

# Peak accessibility statistics
try:
    peak_counts_mat = adata.X.sum(axis=1)  # type: ignore[call-arg]
    if hasattr(peak_counts_mat, 'A1'):
        peak_counts = peak_counts_mat.A1
    else:
        peak_counts = np.asarray(peak_counts_mat).ravel()
except Exception:
    peak_counts = np.asarray(adata.X).sum(axis=1)

adata.obs['n_peaks'] = peak_counts

print_info(f"Peak accessibility: mean={peak_counts.mean():.1f} ± {peak_counts.std():.1f}")
print()

# ==================================================
# Train with Neural ODE
# ==================================================

print_section("Training iAODE with Neural ODE")
print_info("Configuration:")
print("  use_ode=True   → enable Neural ODE")
print("  i_dim=2        → ODE bottleneck dim")
print("  latent_dim=32  → final latent dim")
print("  loss_mode='nb' for scATAC-seq counts")
print()

model = iaode.agent(
    adata, layer='counts',
    latent_dim=32, hidden_dim=512,
    use_ode=True,
    i_dim=2,
    encoder_type='mlp',
    loss_mode='nb',  # NB for scATAC-seq counts
    batch_size=128
)

model.fit(epochs=50, patience=20, val_every=5)

metrics = model.get_resource_metrics()
print_success(f"Trained in {metrics['train_time']:.2f}s ({metrics['actual_epochs']} epochs)")
print_info(f"  Peak GPU memory: {metrics['peak_memory_gb']:.3f} GB")
print()

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

# UMAP on iAODE latent space
n_neighbors = 30
sc.pp.neighbors(adata, use_rep='X_iaode', n_neighbors=n_neighbors)
sc.tl.umap(adata, min_dist=0.3)

print_success(f"Latent: {latent.shape}, I-embed: {iembed.shape}")
print_success(f"Pseudotime range: [{pseudotime.min():.3f}, {pseudotime.max():.3f}]")
print()

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
print_info(f"  Mean magnitude: {velocity_magnitude.mean():.5f} ± {velocity_magnitude.std():.5f}")
print()

# Normalized pseudotime for plotting
pseudotime_norm = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
adata.obs['pseudotime_norm'] = pseudotime_norm

# ==================================================
# Multi-Panel Figure
# ==================================================

print_section("Generating multi-panel figure")

# Global style
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

# 3×3 layout
fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(
    3, 3, figure=fig,
    left=0.06, right=0.98,
    top=0.96, bottom=0.06,
    hspace=0.38, wspace=0.35
)

# UMAP coordinates and shared limits
umap_coords = adata.obsm['X_umap']
x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
padding = 0.05
x_range = x_max - x_min
y_range = y_max - y_min
xlim = [x_min - padding * x_range, x_max + padding * x_range]
ylim = [y_min - padding * y_range, y_max + padding * y_range]

def style_umap_ax(ax, xlabel=True, ylabel=True):
    """Consistent styling for UMAP axes."""
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
    """Attach a styled colorbar."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label(label, fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9, width=1.0)
    cbar.outline.set_linewidth(1.2)
    return cbar

# Row 1: static trajectory views
# A: pseudotime
ax1 = fig.add_subplot(gs[0, 0])
scatter1 = ax1.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=pseudotime_norm, cmap='viridis',
    s=15, alpha=0.8, edgecolors='none', rasterized=True
)
style_umap_ax(ax1)
ax1.set_title('A. Pseudotime', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax1, scatter1, 'Pseudotime')

# B: peak accessibility
ax2 = fig.add_subplot(gs[0, 1])
scatter2 = ax2.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=peak_counts, cmap='YlOrRd',
    s=15, alpha=0.8, edgecolors='none', rasterized=True
)
style_umap_ax(ax2)
ax2.set_title('B. Peak accessibility', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax2, scatter2, 'Peak counts')

# C: velocity magnitude
ax3 = fig.add_subplot(gs[0, 2])
scatter3 = ax3.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=velocity_magnitude, cmap='plasma',
    s=15, alpha=0.8, edgecolors='none', rasterized=True
)
style_umap_ax(ax3)
ax3.set_title('C. Velocity magnitude', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax3, scatter3, 'Magnitude')

# Row 2: velocity fields
# D: streamplot (2 columns)
ax4 = fig.add_subplot(gs[1, :2])
scatter4 = ax4.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=pseudotime_norm, cmap='viridis',
    s=12, alpha=0.6, edgecolors='none',
    rasterized=True, zorder=1
)
ax4.streamplot(
    E_grid[0], E_grid[1], V_grid[0], V_grid[1],
    color='#E63946', density=1.8, linewidth=1.5,
    arrowsize=1.8, arrowstyle='->', zorder=2
)
style_umap_ax(ax4)
ax4.set_title('D. Velocity field (streamplot)', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax4, scatter4, 'Pseudotime', size="1.5%", pad=0.05)

# E: quiver
ax5 = fig.add_subplot(gs[1, 2])
scatter5 = ax5.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=pseudotime_norm, cmap='viridis',
    s=12, alpha=0.6, edgecolors='none',
    rasterized=True, zorder=1
)
step = max(1, adata.n_obs // 150)
ax5.quiver(
    umap_coords[::step, 0], umap_coords[::step, 1],
    velocity_umap[::step, 0], velocity_umap[::step, 1],
    color='#E63946', alpha=0.8, width=0.004,
    headwidth=4.5, headlength=5.5,
    scale=None, scale_units='xy', zorder=2
)
style_umap_ax(ax5)
ax5.set_title('E. Velocity field (quiver)', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax5, scatter5, 'Pseudotime')

# Row 3: distributions + interpretable bottleneck
# F: pseudotime histogram
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(
    pseudotime_norm, bins=40,
    color='#0173B2', alpha=0.75,
    edgecolor='black', linewidth=0.8
)
ax6.axvline(
    pseudotime_norm.mean(), color='#E63946',
    linestyle='--', linewidth=2.5,
    label=f'Mean: {pseudotime_norm.mean():.3f}'
)
ax6.axvline(
    np.median(pseudotime_norm), color='#029E73',
    linestyle='--', linewidth=2.5,
    label=f'Median: {np.median(pseudotime_norm):.3f}'
)
ax6.set_xlabel('Normalized pseudotime', fontsize=11, fontweight='bold')
ax6.set_ylabel('Cell count', fontsize=11, fontweight='bold')
ax6.set_title('F. Pseudotime distribution', fontsize=12, fontweight='bold', loc='left', pad=10)
ax6.legend(
    fontsize=8, frameon=True,
    edgecolor='black', framealpha=0.95,
    loc='upper right'
)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_linewidth(1.2)
ax6.spines['bottom'].set_linewidth(1.2)
ax6.tick_params(width=1.2, labelsize=10)
ax6.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)

# G: velocity magnitude histogram
ax7 = fig.add_subplot(gs[2, 1])
ax7.hist(
    velocity_magnitude, bins=40,
    color='#DE8F05', alpha=0.75,
    edgecolor='black', linewidth=0.8
)
ax7.axvline(
    velocity_magnitude.mean(), color='#E63946',
    linestyle='--', linewidth=2.5,
    label=f'Mean: {velocity_magnitude.mean():.3f}'
)
ax7.axvline(
    np.median(velocity_magnitude), color='#029E73',
    linestyle='--', linewidth=2.5,
    label=f'Median: {np.median(velocity_magnitude):.3f}'
)
ax7.set_xlabel('Velocity magnitude', fontsize=11, fontweight='bold')
ax7.set_ylabel('Cell count', fontsize=11, fontweight='bold')
ax7.set_title('G. Velocity distribution', fontsize=12, fontweight='bold', loc='left', pad=10)
ax7.legend(
    fontsize=8, frameon=True,
    edgecolor='black', framealpha=0.95,
    loc='upper right'
)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.spines['left'].set_linewidth(1.2)
ax7.spines['bottom'].set_linewidth(1.2)
ax7.tick_params(width=1.2, labelsize=10)
ax7.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)

# H: interpretable bottleneck (iembed)
ax8 = fig.add_subplot(gs[2, 2])
scatter8 = ax8.scatter(
    iembed[:, 0], iembed[:, 1],
    c=pseudotime_norm, cmap='viridis',
    s=15, alpha=0.8, edgecolors='none', rasterized=True
)
ax8.set_xlabel('I-embed dim 1', fontsize=11, fontweight='bold')
ax8.set_ylabel('I-embed dim 2', fontsize=11, fontweight='bold')
ax8.set_title('H. Interpretable bottleneck', fontsize=12, fontweight='bold', loc='left', pad=10)
ax8.set_aspect('equal', adjustable='box')
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
print()

# ==================================================
# Summary Statistics
# ==================================================

print_header("Analysis complete")
print_info("Neural ODE trajectory inference summary (scATAC-seq)")
print()
print("  Dataset:")
print(f"    Cells: {adata.n_obs:,}")
print(f"    Peaks (HVP): {adata.n_vars:,}")
print(f"    Mean accessibility: {peak_counts.mean():.1f} ± {peak_counts.std():.1f}")
print()
print("  Training:")
print(f"    Total time: {metrics['train_time']:.2f}s "
      f"({metrics['train_time']/metrics['actual_epochs']:.2f}s/epoch)")
print(f"    Epochs: {metrics['actual_epochs']}")
print(f"    Peak GPU memory: {metrics['peak_memory_gb']:.3f} GB")
print()
print("  Pseudotime:")
print(f"    Range:  [{pseudotime.min():.4f}, {pseudotime.max():.4f}]")
print(f"    Mean:   {pseudotime.mean():.4f} ± {pseudotime.std():.4f}")
print(f"    Median: {np.median(pseudotime):.4f}")
print()

# Velocity statistics
velocity_latent = adata.obsm['X_vf_latent']
velocity_mag_latent = np.linalg.norm(velocity_latent, axis=1)

print("  Velocity:")
print("    Latent space:")
print(f"      Mean magnitude: {velocity_mag_latent.mean():.5f} ± {velocity_mag_latent.std():.5f}")
print(f"      Max magnitude:  {velocity_mag_latent.max():.5f}")
print("    UMAP space:")
print(f"      Mean magnitude: {velocity_magnitude.mean():.5f} ± {velocity_magnitude.std():.5f}")
print(f"      Max magnitude:  {velocity_magnitude.max():.5f}")
print()

# Pseudotime–velocity correlations
from scipy.stats import pearsonr, spearmanr  # type: ignore  # noqa: E402
pearson_corr, pearson_pval = pearsonr(pseudotime_norm, velocity_magnitude)
spearman_corr, spearman_pval = spearmanr(pseudotime_norm, velocity_magnitude)

print("  Pseudotime–velocity correlation:")
print(f"    Pearson:  r = {pearson_corr:.4f} (p = {pearson_pval:.2e})")
print(f"    Spearman: ρ = {spearman_corr:.4f} (p = {spearman_pval:.2e})")
print()

# Peak annotation distribution (if available)
if 'annotation_type' in adata.var.columns:
    print("  Peak annotation:")
    annotation_counts = adata.var['annotation_type'].value_counts()
    for annot, count in annotation_counts.items():
        pct = count / len(adata.var) * 100
        label = str(annot).replace('_', ' ').title()
        print(f"    {label:15s}: {count:5,} ({pct:5.1f}%)")
    print()

print_info("Output files:")
print(f"  • {OUTPUT_DIR}/trajectory_analysis.png")
print(f"  • {OUTPUT_DIR}/trajectory_analysis.pdf")
print()

print_info("Highlights:")
print("  • Trajectory inferred from scATAC-seq via Neural ODE")
print("  • Velocity field in latent and UMAP spaces")
print("  • HVP subset focuses on informative regulatory regions")
print()