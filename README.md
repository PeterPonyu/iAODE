# iAODEVAE: Interpretable Accessibility Ordinary Differential Equation Variational Autoencoder

iAODEVAE is a deep learning framework for single-cell ATAC-seq (scATAC-seq) analysis that combines Variational Autoencoders with Neural Ordinary Differential Equations to model chromatin accessibility dynamics and infer cellular trajectories.

## Overview

iAODEVAE provides comprehensive analysis of scATAC-seq data through:

- **i** (Interpretable): Information bottleneck for interpretable representations
- **A** (ATAC-seq): Optimized for chromatin accessibility data
- **ODE** (Ordinary Differential Equations): Neural ODE integration for temporal dynamics  
- **VAE** (Variational Autoencoder): Probabilistic latent variable modeling

## Key Features

### Core Capabilities
- **Trajectory inference** from chromatin accessibility patterns
- **Data imputation** using learned transition probabilities
- **Velocity field computation** for cellular dynamics visualization
- **Interpretable embeddings** through information bottleneck constraints


## Architecture

The framework consists of modular components:

| Component | Function |
|-----------|----------|
| **Encoder** | Maps peak accessibility to latent space |
| **Decoder** | Reconstructs accessibility from latent representations |
| **Information Bottleneck** | Creates interpretable compressed representations |
| **Neural ODE** | Models continuous dynamics in latent space |

## Installation

```bash
git clone https://github.com/PeterPonyu/iAODE

cd iAODE

pip install -r requirements.txt
```

## Parameters

The `scATACAgent` can be customized with several parameters during initialization. Here are some of the most important ones:

- `adata`: The `AnnData` object containing your scATAC-seq data.
- `layer`: The layer in `adata` to use for training (e.g., "counts").
- `batch_percent`: The fraction of cells to use in each training batch (e.g., `0.1` for 10%).
- `use_ode`: Whether to enable the Neural ODE for trajectory inference.
- `loss_mode`: The statistical model for the data. `"zinb"` (Zero-Inflated Negative Binomial) is recommended for scATAC-seq.
- `latent_dim`: The dimensionality of the main latent space.
- `i_dim`: The dimensionality of the interpretable bottleneck layer.

## Quick Start

### Basic Usage

```python
import scanpy as sc
from iaode import scATACAgent

# Load scATAC-seq data
adata = sc.read_h5ad("scatac_data.h5ad")

# Initialize agent with ODE dynamics
agent = scATACAgent(
    adata, 
    layer="counts",
    batch_percent=0.1,
    use_ode=True,
    loss_mode="zinb"
)

# Train model
agent.fit(epochs=1000)

# Extract representations
representations = agent.get_representations()
latent = representations['latent']
interpretable = representations['interpretable'] 
pseudotime = representations['pseudotime']
```

### Advanced Analysis

```python
# Complete trajectory analysis
results = agent.analyze_trajectory(
    adata,
    latent_key="X_latent",
    embedding_key="X_umap"
)

# Compute velocity field
E_grid, V_grid = agent.compute_velocity_field(
    adata,
    latent_key="X_latent", 
    embedding_key="X_umap"
)

# Data imputation
imputed_data = agent.impute_data(
    top_k=30,
    alpha=0.9
)
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot velocity field
fig, ax = plt.subplots(figsize=(8, 6))
ax.streamplot(E_grid[0], E_grid[1], V_grid[0], V_grid[1], 
             density=1.5, color='black', alpha=0.6)
ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], 
          c=pseudotime, cmap='viridis', s=20)
ax.set_title('Chromatin Accessibility Velocity Field')
plt.show()
```

