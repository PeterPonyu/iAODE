"""
Model evaluation and benchmarking (scRNA-seq – paul15)

Comprehensive comparison of iAODE vs. scVI-family models using
Latent Space Evaluation (LSE) metrics on a trajectory dataset.

Dataset: paul15 (hematopoietic scRNA-seq trajectory)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _example_utils import (
    check_iaode_installed, setup_output_dir,
    print_header, print_section, print_success,
    print_info, print_warning
)

if not check_iaode_installed():
    sys.exit(1)

import iaode
import scanpy as sc  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# Configuration
# ==================================================

CONFIG = {
    'epochs': 100,
    'patience': 20,
    'val_every': 5,
    'latent_dim': 10,
    'hidden_dim': 128,
    'batch_size': 128,
    'test_size': 0.15,
    'val_size': 0.15,
    'random_seed': 42,
}

OUTPUT_DIR = setup_output_dir("model_evaluation")
print_info(f"Saving outputs to: {OUTPUT_DIR}")
print_info(f"Configuration: {CONFIG['epochs']} epochs, latent_dim={CONFIG['latent_dim']}")
print()

# ==================================================
# Load Data
# ==================================================

print_header("Model evaluation & benchmarking (paul15)")

print_section("Loading paul15 dataset")
adata = sc.datasets.paul15()

# Basic QC
sc.pp.filter_cells(adata, min_genes=200)

# Store raw counts BEFORE normalization
from scipy.sparse import issparse  # type: ignore
if issparse(adata.X):
    adata.layers['counts'] = adata.X.copy()
else:
    adata.layers['counts'] = np.asarray(adata.X.copy())

# Normalize for downstream analysis (not used by iAODE loss)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print_success(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
print()

# ==================================================
# Train/Val/Test Split
# ==================================================

print_section("Creating train/validation/test splits")

splitter = iaode.DataSplitter(
    n_samples=adata.n_obs,
    test_size=CONFIG['test_size'],
    val_size=CONFIG['val_size'],
    random_state=CONFIG['random_seed']
)

print_info(
    f"Train: {len(splitter.train_idx)} | "
    f"Val: {len(splitter.val_idx)} | "
    f"Test: {len(splitter.test_idx)} cells"
)
print()

# ==================================================
# Train iAODE
# ==================================================

print_section("Training iAODE (Neural ODE)")

model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=int(CONFIG['latent_dim']),
    hidden_dim=int(CONFIG['hidden_dim']),
    use_ode=True,
    encoder_type='mlp',
    loss_mode='nb',          # Negative binomial for counts
    batch_size=int(CONFIG['batch_size'])
)

model.fit(
    epochs=int(CONFIG['epochs']),
    patience=int(CONFIG['patience']),
    val_every=int(CONFIG['val_every'])
)

latent_iaode = model.get_latent()
metrics_iaode = model.get_resource_metrics()

print_success(
    f"iAODE trained in {metrics_iaode['train_time']:.2f}s "
    f"({metrics_iaode['actual_epochs']} epochs)"
)
print()

# ==================================================
# Evaluate iAODE
# ==================================================

print_section("Evaluating iAODE on test set (LSE metrics)")

latent_iaode_test = latent_iaode[splitter.test_idx]
X_high_test = adata[splitter.test_idx].layers['counts']
if hasattr(X_high_test, 'toarray'):
    X_high_test = X_high_test.toarray()

ls_metrics = iaode.evaluate_single_cell_latent_space(
    latent_space=latent_iaode_test,
    data_type='trajectory',
    verbose=True
)

results = {
    'iAODE': {
        'latent': latent_iaode_test,
        'adata_subset': adata[splitter.test_idx].copy(),
        'train_time': metrics_iaode['train_time'],
        'epochs': metrics_iaode['actual_epochs'],
        'ls_metrics': ls_metrics,
    }
}

print()

# ==================================================
# Train and Evaluate scVI-family Models
# ==================================================

print_section("Training scVI-family models")

scvi_results = iaode.train_scvi_models(
    adata, splitter,
    n_latent=CONFIG['latent_dim'],
    n_epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size']
)

print_section("Evaluating scVI-family models (LSE metrics)")

for model_name, result in scvi_results.items():
    if result is None:
        continue

    print_info(f"Evaluating {model_name.upper()}")

    latent_scvi = result['model'].get_latent_representation(result['adata_test'])

    try:
        ls_scvi = iaode.evaluate_single_cell_latent_space(
            latent_space=latent_scvi,
            data_type='trajectory',
            verbose=False
        )
    except Exception as e:
        print_warning(f"LSE metrics failed for {model_name}: {e}")
        ls_scvi = {}

    results[model_name] = {
        'latent': latent_scvi,
        'adata_subset': result['adata_test'].copy(),
        'train_time': result['train_time'],
        'epochs': result.get('epochs', CONFIG['epochs']),
        'ls_metrics': ls_scvi,
    }

    print_success(f"{model_name.upper()} evaluated")
print()

# ==================================================
# Build Comparison Table
# ==================================================

print_section("Building comparison table")

comparison_data = []
for model_name, data in results.items():
    row = {
        'Model': model_name,
        'Train Time (s)': data['train_time'],
        'Epochs': data['epochs'],
    }

    ls = data.get('ls_metrics', {})
    row['Manifold Dim'] = ls.get('manifold_dimensionality', np.nan)
    row['Spectral Decay'] = ls.get('spectral_decay_rate', np.nan)
    row['Trajectory Dir'] = ls.get('trajectory_directionality', np.nan)

    comparison_data.append(row)

df = pd.DataFrame(comparison_data)
print()
print(df.to_string(index=False))
print()

csv_path = OUTPUT_DIR / 'model_comparison.csv'
df.to_csv(csv_path, index=False)
print_success(f"Saved table: {csv_path}")
print()

# ==================================================
# UMAP for Visualization
# ==================================================

print_section("Computing UMAP embeddings from latent spaces")

for model_name, data in results.items():
    adata_viz = data['adata_subset'].copy()
    adata_viz.obsm['X_latent'] = data['latent']
    sc.pp.neighbors(adata_viz, use_rep='X_latent', n_neighbors=15)
    sc.tl.umap(adata_viz, min_dist=0.3)
    results[model_name]['adata_viz'] = adata_viz

print_success("Computed UMAP for all models")
print()

# ==================================================
# Publication-Quality Figure
# ==================================================

print_section("Generating publication-quality figure")

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

# Colorblind-friendly palette
MODEL_COLORS = {
    'iAODE': '#0173B2',
    'scvi': '#DE8F05',
    'scanvi': '#029E73',
    'peakvi': '#CC78BC',
    'poissonvi': '#CA9161',
}

# Determine which models are available
model_names_ordered = ['iAODE', 'scvi', 'scanvi', 'peakvi', 'poissonvi']
model_names_plot = [m for m in model_names_ordered if m in results]

# 2×4 layout (top row: metrics; bottom row: UMAPs)
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(
    2, 4, figure=fig,
    left=0.06, right=0.98,
    top=0.95, bottom=0.12,
    hspace=0.35, wspace=0.35
)

def plot_metric_bar(ax, df_data, metric_col, title, ylabel='Score', ylim=None):
    """Consistent bar plots for numeric metrics."""
    data_plot = df_data[df_data[metric_col].notna()].copy()

    if data_plot.empty:
        ax.text(0.5, 0.5, 'No data',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.axis('off')
        return

    x_pos = np.arange(len(data_plot))
    colors = [
        MODEL_COLORS.get(m, MODEL_COLORS.get(m.lower(), '#888888'))
        for m in data_plot['Model']
    ]

    bars = ax.bar(
        x_pos, data_plot[metric_col],
        color=colors, edgecolor='black',
        linewidth=1.0, alpha=0.85
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        data_plot['Model'],
        rotation=0, ha='center', fontweight='bold'
    )
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=10)

    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold'
            )

# Prepare numeric DataFrame
df_plot = df.copy()
numeric_cols = ['Manifold Dim', 'Spectral Decay', 'Trajectory Dir', 'Train Time (s)']
for col in numeric_cols:
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

# Row 1: LSE metrics + training time
plot_metric_bar(
    fig.add_subplot(gs[0, 0]),
    df_plot, 'Manifold Dim',
    'A. Manifold dimensionality',
    ylabel='Score', ylim=(0, 1)
)
plot_metric_bar(
    fig.add_subplot(gs[0, 1]),
    df_plot, 'Spectral Decay',
    'B. Spectral decay rate',
    ylabel='Score', ylim=(0, 1)
)
plot_metric_bar(
    fig.add_subplot(gs[0, 2]),
    df_plot, 'Trajectory Dir',
    'C. Trajectory directionality',
    ylabel='Score', ylim=(0, 1)
)

# Panel D: training time
ax_time = fig.add_subplot(gs[0, 3])
df_time = df_plot[df_plot['Train Time (s)'].notna()].copy()

if not df_time.empty:
    x_pos = np.arange(len(df_time))
    colors = [
        MODEL_COLORS.get(m, MODEL_COLORS.get(m.lower(), '#888888'))
        for m in df_time['Model']
    ]

    bars = ax_time.bar(
        x_pos, df_time['Train Time (s)'],
        color=colors, edgecolor='black',
        linewidth=1.0, alpha=0.85
    )

    ax_time.set_xticks(x_pos)
    ax_time.set_xticklabels(
        df_time['Model'],
        rotation=0, ha='center', fontweight='bold'
    )
    ax_time.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_time.set_title('D. Training time', fontsize=12, fontweight='bold', loc='left', pad=10)

    ax_time.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)
    ax_time.spines['left'].set_linewidth(1.2)
    ax_time.spines['bottom'].set_linewidth(1.2)

    for bar in bars:
        height = bar.get_height()
        ax_time.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{height:.1f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )
else:
    ax_time.text(
        0.5, 0.5, 'No timing data',
        ha='center', va='center',
        transform=ax_time.transAxes, fontsize=11
    )
    ax_time.axis('off')

# Row 2: UMAPs colored by paul15 clusters (shared legend)
umap_panels = ['E', 'F', 'G', 'H']
legend_artists = []
legend_labels = []

for idx, (model_name, panel_label) in enumerate(zip(model_names_plot[:4], umap_panels)):
    ax = fig.add_subplot(gs[1, idx])

    if model_name not in results or 'adata_viz' not in results[model_name]:
        ax.text(
            0.5, 0.5,
            f'{model_name.upper()}\nno UMAP',
            ha='center', va='center',
            fontsize=11, color='#666666', style='italic'
        )
        ax.set_title(
            f'{panel_label}. {model_name.upper()}',
            fontsize=12, fontweight='bold', loc='left', pad=10
        )
        ax.axis('off')
        continue

    adata_viz = results[model_name]['adata_viz']
    umap_coords = adata_viz.obsm['X_umap']

    if 'paul15_clusters' in adata_viz.obs.columns:
        # Ensure categorical
        if not pd.api.types.is_categorical_dtype(adata_viz.obs['paul15_clusters']):
            adata_viz.obs['paul15_clusters'] = adata_viz.obs['paul15_clusters'].astype('category')

        categories = adata_viz.obs['paul15_clusters'].cat.categories
        n_cats = len(categories)

        if n_cats <= 10:
            colors_palette = plt.colormaps['tab10'](np.linspace(0,1,10))[:n_cats]
        elif n_cats <= 20:
            colors_palette = plt.colormaps['tab20'](np.linspace(0,1,20))[:n_cats]
        else:
            colors_palette = plt.colormaps['gist_ncar'](np.linspace(0,1,n_cats))

        for i, cat in enumerate(categories):
            mask = adata_viz.obs['paul15_clusters'] == cat
            scatter = ax.scatter(
                umap_coords[mask, 0], umap_coords[mask, 1],
                c=[colors_palette[i]],
                s=8, alpha=0.7,
                edgecolors='none', rasterized=True
            )

            # Collect legend handles from first UMAP only
            if idx == 0:
                legend_artists.append(scatter)
                label = str(cat)
                if len(label) > 12:
                    label = label[:11] + '…'
                legend_labels.append(label)
    else:
        # No clusters: plot all cells in one color
        ax.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c='#4C72B0',
            s=8, alpha=0.7,
            edgecolors='none', rasterized=True
        )

    ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
    ax.set_title(
        f'{panel_label}. {model_name.upper()}',
        fontsize=12, fontweight='bold', loc='left', pad=10
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(labelsize=10)

# Shared legend beneath the figure
if legend_artists:
    n_items = len(legend_labels)
    n_cols = min(n_items, 8)

    fig.legend(
        legend_artists, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=n_cols,
        frameon=True,
        fontsize=8,
        markerscale=2.5,
        columnspacing=1.2,
        handletextpad=0.4,
        edgecolor='black',
        framealpha=0.95
    )

# Save figure
fig_path_png = OUTPUT_DIR / 'model_comparison.png'
fig_path_pdf = OUTPUT_DIR / 'model_comparison.pdf'
plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
plt.close()

print_success(f"Saved figure: {fig_path_png}")
print_success(f"Saved figure: {fig_path_pdf}")
print()

# ==================================================
# Summary
# ==================================================

print_header("Evaluation complete")
print_info("Model benchmarking summary – latent space evaluation")
print()
print(f"  Configuration: {CONFIG['epochs']} epochs, latent_dim={CONFIG['latent_dim']}")
print(f"  Dataset: {adata.n_obs:,} cells (test: {len(splitter.test_idx):,})")
print()
print("  Performance (test latent space):")
for model_name in model_names_plot:
    data = results[model_name]
    metrics_str = []
    if 'ls_metrics' in data:
        m = data['ls_metrics']
        if 'manifold_dimensionality' in m:
            metrics_str.append(f"ManifoldDim={m['manifold_dimensionality']:.3f}")
        if 'spectral_decay_rate' in m:
            metrics_str.append(f"SpectralDecay={m['spectral_decay_rate']:.3f}")
        if 'trajectory_directionality' in m:
            metrics_str.append(f"TrajDir={m['trajectory_directionality']:.3f}")
    print(f"    {model_name.upper():12s}: {data['train_time']:6.1f}s, {', '.join(metrics_str)}")
print()

print_info("Output files:")
print(f"  • {OUTPUT_DIR}/model_comparison.csv")
print(f"  • {OUTPUT_DIR}/model_comparison.png")
print(f"  • {OUTPUT_DIR}/model_comparison.pdf")
print()
print_info("All scores computed using LSE (Latent Space Evaluation) on the test latent spaces.")
print()