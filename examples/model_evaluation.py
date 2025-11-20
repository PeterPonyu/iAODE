"""
Model Evaluation and Benchmarking Example

This example demonstrates how to evaluate iAODE models and compare
them against state-of-the-art methods like scVI.
"""

import anndata as ad
import scanpy as sc
import iaode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
print("Loading data...")
adata = sc.datasets.paul15()

# Preprocess
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")

# ============================================================================
# Part 1: Train iAODE Model
# ============================================================================

print("\n" + "="*70)
print("Training iAODE Model")
print("="*70)

model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=10,
    hidden_dim=128,
    use_ode=True,
    loss_mode='nb',
    encoder_type='mlp',
    batch_size=128
)

model.fit(epochs=100, patience=20, val_every=5)
# get_latent() returns full dataset representation
latent_iaode = model.get_latent()

# ============================================================================
# Part 2: Evaluate Dimensionality Reduction Quality
# ============================================================================

print("\n" + "="*70)
print("Evaluating Dimensionality Reduction Quality")
print("="*70)

# Evaluate against original high-dimensional space
dr_metrics = iaode.evaluate_dimensionality_reduction(
    X_high=adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
    X_low=latent_iaode,
    k=10,
    verbose=True
)

# ============================================================================
# Part 3: Evaluate Latent Space Quality
# ============================================================================

print("\n" + "="*70)
print("Evaluating Latent Space Quality")
print("="*70)

ls_metrics = iaode.evaluate_single_cell_latent_space(
    latent_space=latent_iaode,
    data_type='trajectory',  # Hematopoiesis is a trajectory dataset
    verbose=True
)

# ============================================================================
# Part 4: Benchmark Against scVI Models
# ============================================================================

print("\n" + "="*70)
print("Benchmarking Against scVI Models")
print("="*70)

# Create data splitter (same splits for fair comparison)
splitter = iaode.DataSplitter(
    n_samples=adata.n_obs,
    test_size=0.15,
    val_size=0.15,
    random_state=42
)

# Train scVI models
scvi_results = iaode.train_scvi_models(
    adata,
    splitter,
    n_latent=10,
    n_epochs=100,
    batch_size=128
)

# Evaluate scVI models
scvi_metrics = iaode.evaluate_scvi_models(
    scvi_results,
    adata,
    splitter.test_idx
)

# ============================================================================
# Part 5: Compare All Methods
# ============================================================================

print("\n" + "="*70)
print("Comparing All Methods")
print("="*70)

# Collect results
results = {
    'iAODE': {
        'latent': latent_iaode,
        'train_time': model.get_resource_metrics()['train_time'],
        'dr_metrics': dr_metrics,
        'ls_metrics': ls_metrics
    }
}

# Add scVI results
for model_name, result in scvi_results.items():
    if result is not None:
        # scVI models need adata to be setup before getting latent representation
        # Use only test set for fair comparison (same as metrics evaluation)
        results[model_name] = {
            'latent': result['model'].get_latent_representation(result['adata_test']),
            'adata_subset': result['adata_test'].copy(),  # Store subset for visualization
            'train_time': result['train_time'],
            'metrics': scvi_metrics.get(model_name, {})
        }

# Create comparison DataFrame
comparison_data = []
for model_name, data in results.items():
    row = {'Model': model_name}
    
    # Training metrics
    row['Train Time (s)'] = data['train_time']
    
    if model_name == 'iAODE':
        # DR metrics
        row['Distance Corr'] = data['dr_metrics']['distance_correlation']
        row['Q_local'] = data['dr_metrics']['Q_local']
        row['Q_global'] = data['dr_metrics']['Q_global']
        
        # LS metrics
        row['Manifold Dim'] = data['ls_metrics']['manifold_dimensionality']
        row['Spectral Decay'] = data['ls_metrics']['spectral_decay_rate']
        row['Trajectory Dir'] = data['ls_metrics']['trajectory_directionality']
    else:
        # scVI metrics
        if 'metrics' in data and data['metrics']:
            row['NMI'] = data['metrics'].get('NMI', np.nan)
            row['ARI'] = data['metrics'].get('ARI', np.nan)
            row['ASW'] = data['metrics'].get('ASW', np.nan)
    
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)

print("\n📊 Model Comparison:")
print(df.to_string(index=False))

# Save results
df.to_csv('model_comparison.csv', index=False)
print("\n💾 Results saved to 'model_comparison.csv'")

# ============================================================================
# Part 6: Visualization
# ============================================================================

print("\nCreating visualization...")

fig = plt.figure(figsize=(15, 10))

# Plot 1: Training time comparison
ax1 = plt.subplot(2, 3, 1)
models = df['Model'].values
times = df['Train Time (s)'].values
colors = ['#e74c3c' if m == 'iAODE' else '#3498db' for m in models]
ax1.barh(models, times, color=colors)
ax1.set_xlabel('Training Time (seconds)')
ax1.set_title('Training Time Comparison')

# Plot 2: Distance correlation (iAODE only)
ax2 = plt.subplot(2, 3, 2)
if 'Distance Corr' in df.columns:
    ax2.bar(['Distance Corr', 'Q_local', 'Q_global'], 
            [df.loc[df['Model']=='iAODE', 'Distance Corr'].values[0],
             df.loc[df['Model']=='iAODE', 'Q_local'].values[0],
             df.loc[df['Model']=='iAODE', 'Q_global'].values[0]],
            color='#e74c3c')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Score')
    ax2.set_title('iAODE DR Quality Metrics')

# Plot 3: Latent space quality (iAODE only)
ax3 = plt.subplot(2, 3, 3)
if 'Manifold Dim' in df.columns:
    ax3.bar(['Manifold\nDim', 'Spectral\nDecay', 'Trajectory\nDir'], 
            [df.loc[df['Model']=='iAODE', 'Manifold Dim'].values[0],
             df.loc[df['Model']=='iAODE', 'Spectral Decay'].values[0],
             df.loc[df['Model']=='iAODE', 'Trajectory Dir'].values[0]],
            color='#e74c3c')
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('Score')
    ax3.set_title('iAODE LS Quality Metrics')

# Plot 4-6: UMAP visualizations for different methods
for idx, model_name in enumerate(['iAODE', 'scvi', 'peakvi'], start=4):
    if model_name in results:
        ax = plt.subplot(2, 3, idx)
        latent = results[model_name]['latent']
        
        # Use appropriate adata subset for each model
        if model_name == 'iAODE':
            adata_viz = adata.copy()
        else:
            # scVI models: use test set subset
            adata_viz = results[model_name]['adata_subset'].copy()
        
        # Ensure latent is 2D array
        if latent.ndim == 1:
            latent = latent.reshape(-1, 1)
        adata_viz.obsm['X_latent'] = latent
        sc.pp.neighbors(adata_viz, use_rep='X_latent')
        sc.tl.umap(adata_viz)
        
        if 'paul15_clusters' in adata_viz.obs.columns:
            sc.pl.umap(adata_viz, color='paul15_clusters', ax=ax, show=False)
        ax.set_title(f'{model_name.upper()} Latent Space')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("💾 Visualization saved to 'model_evaluation.png'")

print("\n✅ Evaluation complete!")
