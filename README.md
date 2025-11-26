# iAODE: Interpretable Analysis of Omics Data Explorer

iAODE is a deep learning framework for single-cell omics analysis that combines Variational Autoencoders with Neural Ordinary Differential Equations to model cellular dynamics and infer cellular trajectories. It supports both single-cell ATAC-seq (scATAC-seq) and single-cell RNA-seq (scRNA-seq) data.

## Overview

iAODE provides comprehensive analysis of single-cell omics data through:

- **i** (Interpretable): Information bottleneck for interpretable representations
- **A** (Analysis): Multi-modal analysis supporting scATAC-seq and scRNA-seq data
- **ODE** (Ordinary Differential Equations): Neural ODE integration for temporal dynamics  
- **VAE** (Variational Autoencoder): Probabilistic latent variable modeling

## Key Features

### Core Capabilities
- **Multi-modal Support**: Process both scRNA-seq and scATAC-seq data with specialized loss functions (MSE, NB, ZINB)
- **Trajectory inference** from gene expression or chromatin accessibility patterns
- **Data imputation** using learned transition probabilities
- **Velocity field computation** for cellular dynamics visualization
- **Interpretable embeddings** through information bottleneck constraints
- **GPU-accelerated training** with early stopping and validation monitoring

## Architecture

The framework consists of modular components:

| Component | Function |
|-----------|----------|
| **Encoder** | Maps features (genes/peaks) to latent space |
| **Decoder** | Reconstructs data from latent representations |
| **Information Bottleneck** | Creates interpretable compressed representations |
| **Neural ODE** | Models continuous dynamics in latent space |

## Installation

```bash
git clone https://github.com/PeterPonyu/iAODE

cd iAODE

pip install -r requirements.txt
```

## Parameters

The `agent` can be customized with several parameters during initialization:

- `adata`: The `AnnData` object containing your single-cell data
- `layer`: The layer in `adata` to use for training (e.g., "counts" or "X")
- `use_ode`: Whether to enable the Neural ODE for trajectory inference
- `loss_mode`: The statistical model for the data:
  - `"mse"`: Mean Squared Error (general purpose)
  - `"nb"`: Negative Binomial (recommended for scRNA-seq)
  - `"zinb"`: Zero-Inflated Negative Binomial (recommended for scATAC-seq)
- `latent_dim`: The dimensionality of the main latent space
- `i_dim`: The dimensionality of the interpretable bottleneck layer
- `batch_size`: Training batch size
- `lr`: Learning rate for the optimizer

## Quick Start

### scRNA-seq Analysis

```python
import scanpy as sc
from iaode import agent

# Load scRNA-seq data
adata = sc.read_h5ad("scrna_data.h5ad")

# Initialize agent
model = agent(
    adata, 
    layer="X",
    use_ode=True,
    loss_mode="nb",  # Negative Binomial for RNA
    latent_dim=10,
    i_dim=2
)

# Train model
model.fit(epochs=100, patience=20)

# Extract representations
latent = model.get_latent()
interpretable = model.get_iembed()
```

### scATAC-seq Analysis

```python
import scanpy as sc
from iaode import agent

# Load scATAC-seq data
adata = sc.read_h5ad("scatac_data.h5ad")

# Initialize agent with ZINB loss for sparse ATAC data
model = agent(
    adata, 
    layer="counts",
    use_ode=True,
    loss_mode="zinb",  # Zero-Inflated NB for ATAC
    latent_dim=10,
    i_dim=2
)

# Train model
model.fit(epochs=100, patience=20)

# Extract representations
latent = model.get_latent()
interpretable = model.get_iembed()
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
ax.set_title('Cellular Velocity Field')
plt.show()
```

## Web Interface

iAODE includes several web-based tools:

- **Dataset Browser**: Browse and explore datasets from NCBI GEO
- **Training UI**: Upload data, configure preprocessing, and train models
- **Continuity Explorer**: Interactive exploration of trajectory structures across embedding methods

