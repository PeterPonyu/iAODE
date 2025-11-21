"""
Model Evaluation and Benchmarking - scATAC-seq

Comprehensive evaluation comparing iAODE against scVI-family models
focusing on Latent Space Evaluation metrics.

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
import scanpy as sc  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
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
    'n_hvp': 20000,  # Number of highly variable peaks
}

OUTPUT_DIR = setup_output_dir("model_evaluation_atac")
print_info(f"Outputs saved to: {OUTPUT_DIR}")
print_info(f"Configuration: {CONFIG['epochs']} epochs, latent_dim={CONFIG['latent_dim']}, HVP={CONFIG['n_hvp']}")
print()

# ==================================================
# Load and Annotate scATAC Data
# ==================================================

print_header("Model Evaluation & Benchmarking - scATAC-seq")

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
    n_top_peaks=int(CONFIG['n_hvp'])
)

print_success(f"Annotated: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")

# ==================================================
# Subset to HVPs Only
# ==================================================

print_section("Subsetting to highly variable peaks (HVPs)")

if 'highly_variable' in adata.var.columns:
    n_total_peaks = adata.n_vars
    n_hvp = adata.var['highly_variable'].sum()
    
    # Subset to HVPs
    adata = adata[:, adata.var['highly_variable']].copy()
    
    print_success(f"HVP subset: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
    print_info(f"  Retained {n_hvp:,} / {n_total_peaks:,} peaks ({n_hvp/n_total_peaks*100:.1f}%)")
else:
    print_warning("No HVP information found, using all peaks")

print()

# Ensure counts layer exists
if 'counts' not in adata.layers:
    from scipy.sparse import issparse  # type: ignore
    if issparse(adata.X):
        adata.layers['counts'] = adata.X.copy()
    else:
        adata.layers['counts'] = np.asarray(adata.X.copy())

# Compute peak statistics for visualization
try:
    peak_counts_mat = adata.X.sum(axis=1)  # type: ignore[call-arg]
    if hasattr(peak_counts_mat, 'A1'):
        peak_counts = peak_counts_mat.A1
    else:
        peak_counts = np.asarray(peak_counts_mat).ravel()
except Exception:
    peak_counts = np.asarray(adata.X).sum(axis=1)

adata.obs['n_peaks'] = peak_counts

# ==================================================
# Create Train/Val/Test Split
# ==================================================

print_section("Creating train/validation/test splits")

splitter = iaode.DataSplitter(
    n_samples=adata.n_obs,
    test_size=CONFIG['test_size'],
    val_size=CONFIG['val_size'],
    random_state=CONFIG['random_seed']
)

print_info(f"Train: {len(splitter.train_idx)} | Val: {len(splitter.val_idx)} | Test: {len(splitter.test_idx)} cells")
print()

# ==================================================
# Train iAODE
# ==================================================

print_section("Training iAODE model")

model = iaode.agent(
    adata, 
    layer='counts',
    latent_dim=int(CONFIG['latent_dim']),
    hidden_dim=int(CONFIG['hidden_dim']),
    use_ode=True,
    encoder_type='mlp',
    loss_mode='mse',  # MSE for TF-IDF normalized scATAC data
    batch_size=int(CONFIG['batch_size'])
)

model.fit(
    epochs=int(CONFIG['epochs']),
    patience=int(CONFIG['patience']),
    val_every=int(CONFIG['val_every'])
)

latent_iaode = model.get_latent()
metrics_iaode = model.get_resource_metrics()

print_success(f"iAODE trained in {metrics_iaode['train_time']:.2f}s ({metrics_iaode['actual_epochs']} epochs)")
print_info(f"  Peak GPU memory: {metrics_iaode['peak_memory_gb']:.3f} GB")
print()

# ==================================================
# Evaluate iAODE
# ==================================================

print_section("Evaluating iAODE on test set")

latent_iaode_test = latent_iaode[splitter.test_idx]
X_high_test = adata[splitter.test_idx].layers['counts']
if hasattr(X_high_test, 'toarray'):
    X_high_test = X_high_test.toarray()

# Latent Space Evaluation metrics
print_info("Computing Latent Space Evaluation (LSE) metrics...")
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
        'ls_metrics': ls_metrics
    }
}

print()

# ==================================================
# Train and Evaluate scVI Models
# ==================================================

print_section("Training scVI-family models")

scvi_results = iaode.train_scvi_models(
    adata, splitter,
    n_latent=CONFIG['latent_dim'],
    n_epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size']
)

print_section("Evaluating scVI-family models")

for model_name, result in scvi_results.items():
    if result is not None:
        print_info(f"Evaluating {model_name.upper()}")
        
        latent_scvi = result['model'].get_latent_representation(result['adata_test'])
        
        # Latent Space Evaluation metrics
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
            'ls_metrics': ls_scvi
        }
        
        print_success(f"{model_name.upper()} evaluated")

print()

# ==================================================
# Create Comparison Table
# ==================================================

print_section("Generating comparison table")

comparison_data = []
for model_name, data in results.items():
    row = {
        'Model': model_name,
        'Train Time (s)': data['train_time'],
        'Epochs': data['epochs']
    }
    
    # Latent Space Evaluation metrics
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
print_success(f"Saved: {csv_path}")
print()

# ==================================================
# Compute UMAP for Visualization
# ==================================================

print_section("Computing UMAP embeddings")

for model_name, data in results.items():
    adata_viz = data['adata_subset'].copy()
    adata_viz.obsm['X_latent'] = data['latent']
    sc.pp.neighbors(adata_viz, use_rep='X_latent', n_neighbors=15)
    sc.tl.umap(adata_viz, min_dist=0.3)
    results[model_name]['adata_viz'] = adata_viz
    print_info(f"  {model_name.upper()} UMAP computed")

print()

# ==================================================
# Generate Multi-Panel Figure
# ==================================================

print_section("Generating multi-panel figure")

# Set global style with Ubuntu font
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

# Professional color palette (colorblind-friendly)
MODEL_COLORS = {
    'iAODE': '#0173B2',
    'scvi': '#DE8F05',
    'scanvi': '#029E73',
    'peakvi': '#CC79A7',
    'poissonvi': '#CA9161'
}

# Determine available models - include all trained models up to 5
model_names_ordered = ['iAODE', 'peakvi', 'poissonvi', 'scvi', 'scanvi']
model_names_plot = [m for m in model_names_ordered if m in results.keys()][:5]

n_models = len(model_names_plot)
print_info(f"Visualizing {n_models} models: {', '.join([m.upper() for m in model_names_plot])}")

# Create figure with organized 3-row layout (expand to 5 columns if needed)
fig = plt.figure(figsize=(3.2 * n_models, 12))
gs = gridspec.GridSpec(3, n_models, figure=fig,
                       left=0.07, right=0.98,
                       top=0.96, bottom=0.06,
                       hspace=0.30, wspace=0.30)

def style_axis(ax, grid=True):
    """Apply consistent styling to axes"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)
    if grid:
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)

# Helper function for bar plots
def plot_metric_bar(ax, df_data, metric_col, title, ylabel='Score', ylim=None):
    """Create consistent bar plots for metrics"""
    data_plot = df_data[df_data[metric_col].notna()].copy()
    
    if data_plot.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, color='#666666', style='italic')
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.axis('off')
        return
    
    x_pos = np.arange(len(data_plot))
    colors = [MODEL_COLORS.get(m, MODEL_COLORS.get(m.lower(), '#888888')) for m in data_plot['Model']]
    
    bars = ax.bar(x_pos, data_plot[metric_col], 
                 color=colors, edgecolor='black', 
                 linewidth=1.2, alpha=0.85, width=0.7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(data_plot['Model'], rotation=0, ha='center', fontweight='bold', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=10)
    style_axis(ax, grid=True)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', 
                   va='bottom' if height >= 0 else 'top', 
                   fontsize=9, fontweight='bold')

# Prepare numeric dataframe
df_plot = df.copy()
numeric_cols = ['Manifold Dim', 'Spectral Decay', 'Trajectory Dir', 'Train Time (s)']
for col in numeric_cols:
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

# Row 1: Latent Space Evaluation Metrics (distribute across columns)
metric_cols = ['Manifold Dim', 'Spectral Decay', 'Trajectory Dir', 'Train Time (s)']
metric_titles = [
    'A. Manifold Dimensionality',
    'B. Spectral Decay Rate',
    'C. Trajectory Directionality',
    'D. Training Time'
]
metric_ylabels = ['Score', 'Score', 'Score', 'Time (seconds)']
metric_ylims = [(0, 1), (0, 1), (0, 1), None]

for idx in range(n_models):
    ax = fig.add_subplot(gs[0, idx])
    
    if idx < len(metric_cols):
        col = metric_cols[idx]
        title = metric_titles[idx]
        ylabel = metric_ylabels[idx]
        ylim = metric_ylims[idx]
        
        if col == 'Train Time (s)':
            # Special handling for training time
            df_time = df_plot[df_plot[col].notna()].copy()
            if not df_time.empty:
                x_pos = np.arange(len(df_time))
                colors = [MODEL_COLORS.get(m, MODEL_COLORS.get(m.lower(), '#888888')) for m in df_time['Model']]
                
                bars = ax.bar(x_pos, df_time[col],
                              color=colors, edgecolor='black', 
                              linewidth=1.2, alpha=0.85, width=0.7)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(df_time['Model'], rotation=0, ha='center', fontweight='bold', fontsize=10)
                ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
                ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=10)
                style_axis(ax, grid=True)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', 
                            fontsize=9, fontweight='bold')
        else:
            plot_metric_bar(ax, df_plot, col, title, ylabel=ylabel, ylim=ylim)
    else:
        # Empty panel for excess columns
        ax.axis('off')

# Helper function for UMAP styling
def style_umap_ax(ax, xlim=None, ylim=None):
    """Apply consistent styling to UMAP axes"""
    ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
    if lim_x is not None:
        ax.set_xlim(lim_x)
    if lim_y is not None:
        ax.set_ylim(lim_y)
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)

def add_colorbar(fig, ax, scatter, label):
    """Add consistent colorbar"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label(label, fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9, width=1.0)
    cbar.outline.set_linewidth(1.2)
    return cbar

# Compute consistent color ranges for both visualizations
all_peak_counts = []
all_latent_dim1 = []
for model_name in model_names_plot:
    if model_name in results and 'adata_viz' in results[model_name]:
        all_peak_counts.extend(results[model_name]['adata_viz'].obs['n_peaks'].values)
        all_latent_dim1.extend(results[model_name]['latent'][:, 0])

vmin_peaks = np.percentile(all_peak_counts, 2)
vmax_peaks = np.percentile(all_peak_counts, 98)
vmin_latent = np.percentile(all_latent_dim1, 2)
vmax_latent = np.percentile(all_latent_dim1, 98)

# Compute consistent axis limits
all_x = []
all_y = []
for model_name in model_names_plot:
    if model_name in results and 'adata_viz' in results[model_name]:
        umap_coords = results[model_name]['adata_viz'].obsm['X_umap']
        all_x.extend(umap_coords[:, 0])
        all_y.extend(umap_coords[:, 1])

x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)
padding = 0.05
x_range = x_max - x_min
y_range = y_max - y_min
lim_x: tuple[float, float] = (x_min - padding * x_range, x_max + padding * x_range)
lim_y: tuple[float, float] = (y_min - padding * y_range, y_max + padding * y_range)

# Row 2: UMAP colored by Peak Counts (consistent coloring)
panel_labels_row2 = ['E', 'F', 'G', 'H', 'I']

for idx, model_name in enumerate(model_names_plot):
    ax = fig.add_subplot(gs[1, idx])
    panel_label = panel_labels_row2[idx] if idx < len(panel_labels_row2) else f"P{idx+1}"
    
    if model_name in results and 'adata_viz' in results[model_name]:
        adata_viz = results[model_name]['adata_viz']
        umap_coords = adata_viz.obsm['X_umap']
        color_values = adata_viz.obs['n_peaks'].values
        
        scatter = ax.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=color_values,
            cmap='YlOrRd',
            s=15,
            alpha=0.8,
            edgecolors='none',
            rasterized=True,
            vmin=vmin_peaks,
            vmax=vmax_peaks
        )
        
        style_umap_ax(ax, xlim=lim_x, ylim=lim_y)
        ax.set_title(f'{panel_label}. {model_name.upper()} - Peak Counts',
                    fontsize=11, fontweight='bold', loc='left', pad=8)
        
        # Add colorbar only to the last panel
        if idx == len(model_names_plot) - 1:
            add_colorbar(fig, ax, scatter, 'Peak Counts')
    else:
        ax.text(0.5, 0.5, f'{model_name.upper()}\nnot available',
                ha='center', va='center', fontsize=11,
                color='#666666', style='italic')
        ax.set_title(f'{panel_label}. {model_name.upper()}',
                    fontsize=11, fontweight='bold', loc='left', pad=8)
        ax.axis('off')

# Row 3: UMAP colored by Latent Dimension 1 (consistent coloring)
panel_labels_row3 = ['J', 'K', 'L', 'M', 'N']

for idx, model_name in enumerate(model_names_plot):
    ax = fig.add_subplot(gs[2, idx])
    panel_label = panel_labels_row3[idx] if idx < len(panel_labels_row3) else f"Q{idx+1}"
    
    if model_name in results and 'adata_viz' in results[model_name]:
        adata_viz = results[model_name]['adata_viz']
        umap_coords = adata_viz.obsm['X_umap']
        latent = results[model_name]['latent']
        color_values = latent[:, 0]
        
        scatter = ax.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=color_values,
            cmap='viridis',
            s=15,
            alpha=0.8,
            edgecolors='none',
            rasterized=True,
            vmin=vmin_latent,
            vmax=vmax_latent
        )
        
        style_umap_ax(ax, xlim=lim_x, ylim=lim_y)
        ax.set_title(f'{panel_label}. {model_name.upper()} - Latent Dim 1',
                    fontsize=11, fontweight='bold', loc='left', pad=8)
        
        # Add colorbar only to the last panel
        if idx == len(model_names_plot) - 1:
            add_colorbar(fig, ax, scatter, 'Latent Dim 1')
    else:
        ax.text(0.5, 0.5, f'{model_name.upper()}\nnot available',
                ha='center', va='center', fontsize=11,
                color='#666666', style='italic')
        ax.set_title(f'{panel_label}. {model_name.upper()}',
                    fontsize=11, fontweight='bold', loc='left', pad=8)
        ax.axis('off')

# Save figure
plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()

print_success(f"Saved: {OUTPUT_DIR}/model_comparison.png")
print_success(f"Saved: {OUTPUT_DIR}/model_comparison.pdf")
print()

# ==================================================
# Summary
# ==================================================

print_header("Evaluation Complete")
print_info("Model Benchmarking Summary - scATAC-seq with HVP subset")
print()
print("  Configuration:")
print("    • Dataset: 10X Mouse Brain 5k scATAC-seq")
print(f"    • HVP subset: {adata.n_vars:,} peaks")
print(f"    • Test cells: {len(splitter.test_idx):,}")
print(f"    • Epochs: {CONFIG['epochs']}")
print(f"    • Latent dim: {CONFIG['latent_dim']}")
print()
print("  Performance Summary:")
for model_name in model_names_plot:
    if model_name in results:
        data = results[model_name]
        metrics_str = []
        if 'ls_metrics' in data and 'manifold_dimensionality' in data['ls_metrics']:
            metrics_str.append(f"ManifoldDim={data['ls_metrics']['manifold_dimensionality']:.3f}")
        if 'ls_metrics' in data and 'spectral_decay_rate' in data['ls_metrics']:
            metrics_str.append(f"SpectralDecay={data['ls_metrics']['spectral_decay_rate']:.3f}")
        if 'ls_metrics' in data and 'trajectory_directionality' in data['ls_metrics']:
            metrics_str.append(f"TrajDir={data['ls_metrics']['trajectory_directionality']:.3f}")
        print(f"    {model_name.upper():12s}: {data['train_time']:6.1f}s, {', '.join(metrics_str)}")
print()

print_info("Output files:")
print(f"  • {OUTPUT_DIR}/model_comparison.csv")
print(f"  • {OUTPUT_DIR}/model_comparison.png")
print(f"  • {OUTPUT_DIR}/model_comparison.pdf")
print()

print_info("Visualization Summary:")
print("  • Row 1: Latent Space Evaluation (LSE) metrics + Training time")
print("  • Row 2: UMAP embeddings colored by peak counts (consistent scale)")
print("  • Row 3: UMAP embeddings colored by latent dimension 1 (consistent scale)")
print("  • All UMAPs use same axis limits and color ranges for fair comparison")
print()