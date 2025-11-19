"""
Trajectory Inference Example: Using Neural ODE for Cell Differentiation

This example demonstrates using iAODE with neural ODE enabled
for modeling continuous cellular trajectories.
"""

import anndata as ad
import scanpy as sc
import iaode
import numpy as np
import matplotlib.pyplot as plt

# Load dataset with developmental trajectory
print("Loading data...")
adata = sc.datasets.paul15()  # Hematopoiesis dataset

# Preprocess
print("Preprocessing...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")

# Create model with ODE enabled
print("\nCreating model with Neural ODE...")
model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=10,
    i_dim=2,              # Interpretable embedding dimension
    hidden_dim=128,
    use_ode=True,         # Enable neural ODE
    loss_mode='nb',
    encoder_type='mlp',
    lr=1e-4,
    vae_reg=0.5,         # Weight for VAE latent
    ode_reg=0.5,         # Weight for ODE latent
    batch_size=128
)

# Train
print("\nTraining model...")
model.fit(epochs=100, patience=20, val_every=5)

# Extract representations
print("\nExtracting representations...")
latent = model.get_latent()           # Combined VAE+ODE latent
iembed = model.get_iembed()          # Interpretable embedding
pseudo_time = model.take_time(adata.X)  # Learned pseudo-time

# Add to AnnData
adata.obsm['X_iaode_latent'] = latent
adata.obsm['X_iaode_iembed'] = iembed
adata.obs['iaode_time'] = pseudo_time

# Compute transition matrix for trajectory
print("Computing cell-cell transitions...")
transition_matrix = model.take_transition(adata.X, top_k=30)
adata.obsp['T_iaode'] = transition_matrix

# Visualization
print("\nGenerating visualizations...")

# UMAP on latent space
sc.pp.neighbors(adata, use_rep='X_iaode_latent')
sc.tl.umap(adata)

# Create figure with multiple panels
fig = plt.figure(figsize=(18, 5))

# Panel 1: UMAP colored by cell type
ax1 = plt.subplot(131)
if 'paul15_clusters' in adata.obs.columns:
    sc.pl.umap(adata, color='paul15_clusters', ax=ax1, show=False)
    ax1.set_title('Cell Types')

# Panel 2: UMAP colored by pseudo-time
ax2 = plt.subplot(132)
sc.pl.umap(adata, color='iaode_time', ax=ax2, show=False, cmap='viridis')
ax2.set_title('Learned Pseudo-time')

# Panel 3: Interpretable embedding
ax3 = plt.subplot(133)
scatter = ax3.scatter(
    iembed[:, 0], 
    iembed[:, 1],
    c=pseudo_time,
    cmap='viridis',
    s=10,
    alpha=0.6
)
ax3.set_xlabel('Interpretable Dim 1')
ax3.set_ylabel('Interpretable Dim 2')
ax3.set_title('2D Interpretable Embedding')
plt.colorbar(scatter, ax=ax3, label='Pseudo-time')

plt.tight_layout()
plt.savefig('iaode_trajectory.png', dpi=300, bbox_inches='tight')
print("Trajectory visualization saved to 'iaode_trajectory.png'")

# Velocity-like analysis
print("\nComputing trajectory velocity...")
grads = model.take_grad(adata.X)

# Plot velocity field in 2D embedding space
fig, ax = plt.subplots(figsize=(8, 8))

# Subsample for clearer visualization
n_sample = min(500, len(iembed))
indices = np.random.choice(len(iembed), n_sample, replace=False)

ax.scatter(iembed[:, 0], iembed[:, 1], c='lightgray', s=5, alpha=0.3)
ax.quiver(
    iembed[indices, 0],
    iembed[indices, 1],
    grads[indices, 0],
    grads[indices, 1],
    pseudo_time[indices],
    cmap='viridis',
    scale=20,
    width=0.003,
    alpha=0.7
)
ax.set_xlabel('Interpretable Dim 1')
ax.set_ylabel('Interpretable Dim 2')
ax.set_title('Trajectory Velocity Field')
plt.colorbar(ax.collections[-1], ax=ax, label='Pseudo-time')

plt.tight_layout()
plt.savefig('iaode_velocity.png', dpi=300, bbox_inches='tight')
print("Velocity field saved to 'iaode_velocity.png'")

print("\nDone!")
