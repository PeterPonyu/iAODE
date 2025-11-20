"""
Model Evaluation and Benchmarking

Comprehensive evaluation comparing iAODE against scVI-family models
with consistent metrics across all methods.

Dataset: paul15 (scRNA-seq trajectory)
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
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = setup_output_dir("model_evaluation")
print_info(f"Outputs saved to: {OUTPUT_DIR}")

# ==================================================
# Load Data
# ==================================================

print_header("Model Evaluation & Benchmarking")
print_section("Loading paul15 dataset")

adata = sc.datasets.paul15()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

print_success(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

# ==================================================
# Train iAODE
# ==================================================

print_section("Training iAODE model")

model = iaode.agent(
    adata, layer='counts', latent_dim=10, hidden_dim=128,
    use_ode=True, encoder_type='mlp', loss_mode='nb', batch_size=128
)

model.fit(epochs=400, patience=20, val_every=5)
latent_iaode = model.get_latent()

metrics_iaode = model.get_resource_metrics()
print_success(f"iAODE trained in {metrics_iaode['train_time']:.2f}s")

# ==================================================
# Create Test Set for Fair Comparison
# ==================================================

print_section("Creating test split for fair comparison")

splitter = iaode.DataSplitter(
    n_samples=adata.n_obs,
    test_size=0.15,
    val_size=0.15,
    random_state=42
)

print_info(f"Train: {len(splitter.train_idx)} cells, Test: {len(splitter.test_idx)} cells")

# ==================================================
# Evaluate iAODE (on test set)
# ==================================================

print_section("Evaluating iAODE")

# Get test set data
latent_iaode_test = latent_iaode[splitter.test_idx]
X_high_test = adata[splitter.test_idx].X
if hasattr(X_high_test, 'toarray'):
    X_high_test = X_high_test.toarray()

# Dimensionality Reduction metrics
dr_metrics = iaode.evaluate_dimensionality_reduction(
    X_high=X_high_test,
    X_low=latent_iaode_test,
    k=10,
    verbose=True
)

# Latent Space metrics
ls_metrics = iaode.evaluate_single_cell_latent_space(
    latent_space=latent_iaode_test,
    data_type='trajectory',
    verbose=True
)

# Clustering metrics for iAODE
print_info("Computing clustering metrics for iAODE")
adata_test = adata[splitter.test_idx].copy()
adata_test.obsm['X_latent'] = latent_iaode_test

# Compute neighbors and clustering in latent space
sc.pp.neighbors(adata_test, use_rep='X_latent')
sc.tl.leiden(adata_test, key_added='leiden_latent')

# Compute clustering metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

if 'paul15_clusters' in adata_test.obs.columns:
    true_labels = adata_test.obs['paul15_clusters'].values
    pred_labels = adata_test.obs['leiden_latent'].values
    
    cluster_metrics_iaode = {
        'NMI': normalized_mutual_info_score(true_labels, pred_labels),
        'ARI': adjusted_rand_score(true_labels, pred_labels),
        'ASW': silhouette_score(latent_iaode_test, true_labels)
    }
    print_success(f"iAODE Clustering - NMI: {cluster_metrics_iaode['NMI']:.3f}, ARI: {cluster_metrics_iaode['ARI']:.3f}, ASW: {cluster_metrics_iaode['ASW']:.3f}")
else:
    cluster_metrics_iaode = {}

# ==================================================
# Train scVI Models
# ==================================================

print_section("Training scVI-family models")

scvi_results = iaode.train_scvi_models(
    adata, splitter,
    n_latent=10, n_epochs=400, batch_size=128
)

# Evaluate scVI models with clustering metrics
scvi_metrics = iaode.evaluate_scvi_models(
    scvi_results, adata, splitter.test_idx
)

# ==================================================
# Comprehensive Evaluation for All Models
# ==================================================

print_section("Computing comprehensive metrics for all models")

results = {'iAODE': {
    'latent': latent_iaode_test,
    'adata_subset': adata[splitter.test_idx].copy(),
    'train_time': metrics_iaode['train_time'],
    'dr_metrics': dr_metrics,
    'ls_metrics': ls_metrics,
    'cluster_metrics': cluster_metrics_iaode
}}

# Evaluate scVI models with same metrics as iAODE
for model_name, result in scvi_results.items():
    if result is not None:
        print_info(f"Evaluating {model_name.upper()}")
        
        # Get latent from test set
        latent_scvi = result['model'].get_latent_representation(result['adata_test'])
        
        # Get corresponding high-dim data
        X_high_test = adata[splitter.test_idx].X
        if hasattr(X_high_test, 'toarray'):
            X_high_test = X_high_test.toarray()
        
        # DR metrics for scVI
        try:
            dr_scvi = iaode.evaluate_dimensionality_reduction(
                X_high=X_high_test,
                X_low=latent_scvi,
                k=10,
                verbose=False
            )
        except Exception as e:
            print_warning(f"DR metrics failed for {model_name}: {e}")
            dr_scvi = {}
        
        # LS metrics for scVI
        try:
            ls_scvi = iaode.evaluate_single_cell_latent_space(
                latent_space=latent_scvi,
                data_type='trajectory',
                verbose=False
            )
        except Exception as e:
            print_warning(f"LS metrics failed for {model_name}: {e}")
            ls_scvi = {}
        
        results[model_name] = {
            'latent': latent_scvi,
            'adata_subset': result['adata_test'].copy(),
            'train_time': result['train_time'],
            'dr_metrics': dr_scvi,
            'ls_metrics': ls_scvi,
            'cluster_metrics': scvi_metrics.get(model_name, {})
        }
        
        print_success(f"{model_name.upper()} evaluated")

# ==================================================
# Create Comparison Table
# ==================================================

print_section("Generating comparison table")

comparison_data = []
for model_name, data in results.items():
    row = {'Model': model_name, 'Train Time (s)': data['train_time']}
    
    # DR metrics
    if 'dr_metrics' in data and data['dr_metrics']:
        row['Distance Corr'] = data['dr_metrics'].get('distance_correlation', np.nan)
        row['Q_local'] = data['dr_metrics'].get('Q_local', np.nan)
        row['Q_global'] = data['dr_metrics'].get('Q_global', np.nan)
    else:
        row.update({'Distance Corr': np.nan, 'Q_local': np.nan, 'Q_global': np.nan})
    
    # LS metrics
    if 'ls_metrics' in data and data['ls_metrics']:
        row['Manifold Dim'] = data['ls_metrics'].get('manifold_dimensionality', np.nan)
        row['Spectral Decay'] = data['ls_metrics'].get('spectral_decay_rate', np.nan)
        row['Trajectory Dir'] = data['ls_metrics'].get('trajectory_directionality', np.nan)
    else:
        row.update({'Manifold Dim': np.nan, 'Spectral Decay': np.nan, 'Trajectory Dir': np.nan})
    
    # Clustering metrics
    if 'cluster_metrics' in data and data['cluster_metrics']:
        row['NMI'] = data['cluster_metrics'].get('NMI', np.nan)
        row['ARI'] = data['cluster_metrics'].get('ARI', np.nan)
        row['ASW'] = data['cluster_metrics'].get('ASW', np.nan)
    else:
        row.update({'NMI': np.nan, 'ARI': np.nan, 'ASW': np.nan})
    
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))

csv_path = OUTPUT_DIR / 'model_comparison.csv'
df.to_csv(csv_path, index=False)
print_success(f"Saved: {csv_path}")

# ==================================================
# Visualizations
# ==================================================

print_section("Generating comparison visualizations")

# Use full iAODE latent for main UMAP
adata.obsm['X_iaode'] = latent_iaode
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)

plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 10})

fig = plt.figure(figsize=(18, 15))

# Plot 1-6: Metrics comparison bar plots (2 rows x 3 cols)
metrics_to_plot = [
    ('Distance Corr', 'Distance Correlation'),
    ('Q_local', 'Local Quality'),
    ('Q_global', 'Global Quality'),
    ('NMI', 'Normalized Mutual Info'),
    ('ARI', 'Adjusted Rand Index'),
    ('ASW', 'Silhouette Score')
]

for idx, (col, title) in enumerate(metrics_to_plot, 1):
    ax = plt.subplot(3, 3, idx)
    data_plot = df[df[col].notna()]
    if not data_plot.empty:
        colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(data_plot)]
        ax.bar(data_plot['Model'], data_plot[col], color=colors)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

# Plot 7-9: UMAP visualizations (single legend per row)
legend_handles = None
for idx, model_name in enumerate(['iAODE', 'scvi', 'peakvi'], start=7):
    if model_name in results:
        ax = plt.subplot(3, 3, idx)
        
        # Use the subset data that matches the latent representation
        adata_viz = results[model_name]['adata_subset'].copy()
        latent = results[model_name]['latent']
        
        if latent.ndim == 1:
            latent = latent.reshape(-1, 1)
        
        adata_viz.obsm['X_latent'] = latent
        sc.pp.neighbors(adata_viz, use_rep='X_latent')
        sc.tl.umap(adata_viz)
        
        if 'paul15_clusters' in adata_viz.obs.columns:
            # Plot without legend first
            sc.pl.umap(adata_viz, color='paul15_clusters', ax=ax, show=False, 
                      frameon=True, legend_loc=None)
            
            # Capture legend from first plot only
            if idx == 7:
                legend_handles = ax.get_legend_handles_labels()
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            else:
                # Remove legend from other plots
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        
        ax.set_title(f'{model_name.upper()} Latent Space', fontweight='bold')

# Add single shared legend for UMAP plots
if legend_handles is not None and legend_handles[0]:
    fig.legend(legend_handles[0], legend_handles[1], 
              loc='lower center', ncol=len(legend_handles[0]), 
              bbox_to_anchor=(0.5, -0.02), frameon=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print_success(f"Saved: {OUTPUT_DIR}/model_comparison.png")

print_header("Evaluation Complete")
print_info("All models evaluated with consistent metrics")
