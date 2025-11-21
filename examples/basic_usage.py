"""
Basic Usage Example - scATAC-seq Dimensionality Reduction

This example demonstrates basic iAODE model training for scATAC-seq data
with peak annotation, TF-IDF normalization, and UMAP visualization.

Dataset: 10X Mouse Brain 5k scATAC-seq
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
import scanpy as sc  # type: ignore
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = setup_output_dir("basic_usage")

# ==================================================
# Load and Annotate Data
# ==================================================

print_header("Basic iAODE Usage - scATAC-seq")

print_section("Loading and annotating scATAC-seq data")
print_info("Dataset: 10X Mouse Brain 5k scATAC-seq")
print()

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

print_success(f"Loaded and annotated: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
print()

# ==================================================
# Train Model
# ==================================================

print_section("Training iAODE model")
print_info("Model configuration:")
print("  • Latent dimension: 10")
print("  • Hidden dimension: 128")
print("  • Encoder type: MLP")
print("  • Loss mode: MSE (suitable for TF-IDF normalized peaks)")
print("  • Batch size: 128")
print()

model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=10,
    hidden_dim=128,
    encoder_type='mlp',
    loss_mode='mse',  # MSE for TF-IDF normalized data
    batch_size=128
)

model.fit(epochs=100, patience=20, val_every=5)

metrics = model.get_resource_metrics()
print_success(f"Training complete: {metrics['train_time']:.2f}s ({metrics['actual_epochs']} epochs)")
print_info(f"  Time per epoch: {metrics['train_time']/metrics['actual_epochs']:.2f}s")
print_info(f"  Peak GPU memory: {metrics['peak_memory_gb']:.3f} GB")
print()

# ==================================================
# Extract Latent Representations
# ==================================================

print_section("Extracting latent representations")

latent = model.get_latent()
adata.obsm['X_iaode'] = latent

# Compute UMAP
sc.pp.neighbors(adata, use_rep='X_iaode', n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3)

print_success(f"Latent space: {latent.shape}")
print_info(f"  Mean: {latent.mean():.3f} ± {latent.std():.3f}")
print_info(f"  Range: [{latent.min():.3f}, {latent.max():.3f}]")
print()

# ==================================================
# Visualization
# ==================================================

print_section("Generating visualizations")

# Set global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Ubuntu', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Create figure with 2×2 grid
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.08, right=0.96,
                       top=0.94, bottom=0.08,
                       hspace=0.30, wspace=0.35)

# Get UMAP coordinates
umap_coords = adata.obsm['X_umap']
x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
padding = 0.05
x_range = x_max - x_min
y_range = y_max - y_min
xlim = [x_min - padding * x_range, x_max + padding * x_range]
ylim = [y_min - padding * y_range, y_max + padding * y_range]

def style_umap_ax(ax):
    """Apply consistent styling to UMAP axes"""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)

def add_colorbar(fig, ax, scatter, label):
    """Add consistent colorbar"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label(label, fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9, width=1.0)
    cbar.outline.set_linewidth(1.2)
    return cbar

# Panel A: Latent Dimension 1
ax1 = fig.add_subplot(gs[0, 0])
scatter1 = ax1.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c=latent[:, 0], cmap='viridis',
                       s=12, alpha=0.8, edgecolors='none', rasterized=True)
style_umap_ax(ax1)
ax1.set_title('A. Latent Dimension 1', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax1, scatter1, 'Latent Dim 1')

# Panel B: Latent Dimension 2
ax2 = fig.add_subplot(gs[0, 1])
scatter2 = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c=latent[:, 1], cmap='plasma',
                       s=12, alpha=0.8, edgecolors='none', rasterized=True)
style_umap_ax(ax2)
ax2.set_title('B. Latent Dimension 2', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax2, scatter2, 'Latent Dim 2')

# Panel C: Total Peak Counts
ax3 = fig.add_subplot(gs[1, 0])
try:
    peak_counts_mat = adata.X.sum(axis=1)  # type: ignore[call-arg]
    if hasattr(peak_counts_mat, 'A1'):
        peak_counts = peak_counts_mat.A1
    else:
        peak_counts = np.asarray(peak_counts_mat).ravel()
except Exception:
    peak_counts = np.asarray(adata.X).sum(axis=1)

scatter3 = ax3.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c=peak_counts, cmap='YlOrRd',
                       s=12, alpha=0.8, edgecolors='none', rasterized=True)
style_umap_ax(ax3)
ax3.set_title('C. Total Peak Counts', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax3, scatter3, 'Peak Counts')

# Panel D: Latent Space Variance
ax4 = fig.add_subplot(gs[1, 1])
latent_variance = np.var(latent, axis=1)
scatter4 = ax4.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c=latent_variance, cmap='coolwarm',
                       s=12, alpha=0.8, edgecolors='none', rasterized=True)
style_umap_ax(ax4)
ax4.set_title('D. Latent Space Variance', fontsize=12, fontweight='bold', loc='left', pad=10)
add_colorbar(fig, ax4, scatter4, 'Variance')

# Save figure
plt.savefig(OUTPUT_DIR / 'latent_space_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'latent_space_analysis.pdf', dpi=300, bbox_inches='tight')
plt.close()

print_success(f"Saved: {OUTPUT_DIR}/latent_space_analysis.png")
print_success(f"Saved: {OUTPUT_DIR}/latent_space_analysis.pdf")
print()

# ==================================================
# Additional Analysis: Latent Dimension Distribution
# ==================================================

print_section("Analyzing latent space distribution")

fig2 = plt.figure(figsize=(14, 5))
gs2 = gridspec.GridSpec(1, 3, figure=fig2,
                        left=0.08, right=0.96,
                        top=0.88, bottom=0.15,
                        wspace=0.35)

def style_hist_ax(ax):
    """Apply consistent styling to histogram axes"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)

# Histogram of first 3 latent dimensions
colors = ['#0077BB', '#EE7733', '#009988']
for i in range(3):
    ax = fig2.add_subplot(gs2[0, i])
    ax.hist(latent[:, i], bins=40, color=colors[i], 
            alpha=0.75, edgecolor='black', linewidth=0.8)
    ax.axvline(latent[:, i].mean(), color='#CC3311',
              linestyle='--', linewidth=2.5,
              label=f'Mean: {latent[:, i].mean():.3f}')
    ax.set_xlabel(f'Latent Dimension {i+1}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
    ax.set_title(f'Latent Dim {i+1} Distribution', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, edgecolor='black', framealpha=0.95)
    style_hist_ax(ax)

plt.savefig(OUTPUT_DIR / 'latent_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'latent_distributions.pdf', dpi=300, bbox_inches='tight')
plt.close()

print_success(f"Saved: {OUTPUT_DIR}/latent_distributions.png")
print_success(f"Saved: {OUTPUT_DIR}/latent_distributions.pdf")
print()

# ==================================================
# Summary
# ==================================================

print_header("Analysis Complete")

print_info("Dataset Summary:")
print(f"  Cells: {adata.n_obs:,}")
print(f"  Peaks: {adata.n_vars:,}")
print(f"  Mean peaks per cell: {peak_counts.mean():.1f}")
print()

print_info("Model Performance:")
print(f"  Training time: {metrics['train_time']:.2f}s")
print(f"  Epochs: {metrics['actual_epochs']}")
print(f"  Peak GPU memory: {metrics['peak_memory_gb']:.3f} GB")
print()

print_info("Latent Space Statistics:")
for i in range(min(5, latent.shape[1])):
    print(f"  Dim {i+1}: mean={latent[:, i].mean():7.3f}, std={latent[:, i].std():.3f}")
print()

print_info("Output files:")
print(f"  • {OUTPUT_DIR}/latent_space_analysis.png")
print(f"  • {OUTPUT_DIR}/latent_space_analysis.pdf")
print(f"  • {OUTPUT_DIR}/latent_distributions.png")
print(f"  • {OUTPUT_DIR}/latent_distributions.pdf")
print()

print_header("Next Steps")
print_info("Continue with advanced analyses:")
print("  1. trajectory_inference.py - Neural ODE for trajectory analysis")
print("  2. Clustering analysis using sc.tl.leiden(adata)")
print("  3. Differential accessibility analysis")
print("  4. Peak-to-gene linkage for regulatory analysis")
print()