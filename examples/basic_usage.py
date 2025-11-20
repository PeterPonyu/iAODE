"""
Basic Usage Example: Training iAODE on scRNA-seq Data

This example demonstrates the basic workflow of using iAODE for
dimensionality reduction on single-cell RNA-seq data.
"""

import anndata as ad
import scanpy as sc
import iaode

# Load example dataset (or use your own data)
print("Loading data...")
adata = sc.datasets.paul15()  # Example: mouse hematopoiesis dataset

# Preprocess data (standard scanpy workflow)
print("Preprocessing...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Store raw counts in layer for iAODE
adata.layers['counts'] = adata.X.copy()

print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")

# Create iAODE model
print("\nCreating model...")
model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=10,
    hidden_dim=128,
    use_ode=False,        # Set to True for trajectory inference
    loss_mode='nb',       # Negative binomial for count data
    encoder_type='mlp',
    lr=1e-4,
    batch_size=128,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_seed=42
)

# Train model
print("\nTraining model...")
model.fit(
    epochs=100,
    patience=20,
    val_every=5,
    early_stop=True
)

# Get latent representation
print("\nExtracting latent representation...")
latent = model.get_latent()

# Add to AnnData object
adata.obsm['X_iaode'] = latent

# Visualize with UMAP
print("\nComputing UMAP...")
sc.pp.neighbors(adata, use_rep='X_iaode', n_neighbors=15)
sc.tl.umap(adata)

# Plot results
import matplotlib.pyplot as plt

# Visualize UMAP colored by cell type
if 'paul15_clusters' in adata.obs.columns:
    sc.pl.umap(adata, color='paul15_clusters', title='Cell Types', 
               save='_iaode_celltypes.png', show=False)

# Visualize UMAP colored by latent dimension 1
adata.obs['Latent_Dim1'] = adata.obsm['X_iaode'][:, 0]
sc.pl.umap(adata, color='Latent_Dim1', cmap='viridis', 
           title='Latent Dimension 1', save='_iaode_latent.png', show=False)

print("\nUMAP plots saved to figures/umap_iaode_celltypes.png and figures/umap_iaode_latent.png")

# Print model performance metrics
resource_metrics = model.get_resource_metrics()
print("\nModel Performance:")
print(f"  Training time: {resource_metrics['train_time']:.2f}s")
print(f"  Peak GPU memory: {resource_metrics['peak_memory_gb']:.3f} GB")
print(f"  Actual epochs: {resource_metrics['actual_epochs']}")

print("\nDone!")
