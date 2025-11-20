# Quick Start Guide - iAODE

This guide will help you get started with iAODE in 5 minutes.

## Installation

```bash
pip install iaode
```

## 30-Second Example

```python
import scanpy as sc
import iaode

# Load data
adata = sc.datasets.paul15()

# Preprocess
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

# Train model
model = iaode.agent(adata, layer='counts', latent_dim=10)
model.fit(epochs=50)

# Get results
latent = model.get_latent()

# Visualize
adata.obsm['X_iaode'] = latent
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)
sc.pl.umap(adata, color='paul15_clusters')
```

## Key Features

### 1. Flexible VAE Architecture

```python
# Standard VAE
model = iaode.agent(
    adata,
    latent_dim=10,
    hidden_dim=128,
    encoder_type='mlp',
    loss_mode='nb'
)

# With Transformer encoder
model = iaode.agent(
    adata,
    encoder_type='transformer',
    encoder_num_layers=4,
    encoder_n_heads=8
)

# With multiple loss terms
model = iaode.agent(
    adata,
    recon=1.0,
    beta=1.0,
    tc=0.5,
    dip=0.1
)
```

### 2. Trajectory Inference with Neural ODE

```python
model = iaode.agent(
    adata,
    use_ode=True,
    latent_dim=10,
    i_dim=2
)

model.fit(epochs=100)

# Get trajectory information
latent = model.get_latent()
iembed = model.get_iembed()
pseudo_time = model.take_time(adata.X)
transitions = model.take_transition(adata.X)
```

### 3. scATAC-seq Peak Annotation

```python
adata = iaode.annotation_pipeline(
    h5_file='filtered_peak_bc_matrix.h5',
    gtf_file='gencode.v44.annotation.gtf',
    output_h5ad='annotated_peaks.h5ad',
    promoter_upstream=2000,
    n_top_peaks=20000
)
```

### 4. Model Evaluation

```python
# Dimensionality reduction quality
dr_metrics = iaode.evaluate_dimensionality_reduction(
    X_high=adata.X,
    X_low=latent,
    k=10
)

# Latent space quality
ls_metrics = iaode.evaluate_single_cell_latent_space(
    latent_space=latent,
    data_type='trajectory'
)
```

## Common Use Cases

### Single-cell RNA-seq Analysis

```python
import scanpy as sc
import iaode

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Standard preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

# Train iAODE
model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=10,
    loss_mode='nb',
    use_ode=False
)

model.fit(epochs=100, patience=20)

# Extract latent representation
latent = model.get_latent()
adata.obsm['X_iaode'] = latent

# Downstream analysis (clustering, visualization, etc.)
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

### Developmental Trajectory

```python
# Enable ODE for trajectory
model = iaode.agent(
    adata,
    use_ode=True,
    latent_dim=10,
    i_dim=2,
    vae_reg=0.5,
    ode_reg=0.5
)

model.fit(epochs=100)

# Get trajectory components
latent = model.get_latent()
pseudo_time = model.take_time(adata.X)
velocity = model.take_grad(adata.X)

# Visualize trajectory
adata.obs['pseudo_time'] = pseudo_time
sc.pl.umap(adata, color='pseudo_time', cmap='viridis')
```

### scATAC-seq Analysis

```python
# Annotate peaks
adata = iaode.load_10x_h5_data('filtered_peak_bc_matrix.h5')
adata = iaode.add_peak_coordinates(adata)
adata = iaode.annotate_peaks_to_genes(
    adata,
    gtf_file='gencode.v44.annotation.gtf',
    promoter_upstream=2000
)

# Normalize and select HVPs
iaode.tfidf_normalization(adata, scale_factor=1e4)
iaode.select_highly_variable_peaks(adata, n_top_peaks=20000)

# Train on HVPs
adata_hvp = adata[:, adata.var['highly_variable']].copy()
model = iaode.agent(adata_hvp, layer='counts', latent_dim=10)
model.fit(epochs=100)
```

## Tips and Best Practices

### 1. Choosing Loss Mode
- **MSE**: Normalized/scaled continuous data
- **NB**: Count data (scRNA-seq, scATAC-seq)
- **ZINB**: Count data with high zero-inflation

### 2. Hyperparameter Tuning
```python
# Start with defaults
model = iaode.agent(adata)

# Increase model capacity for complex data
model = iaode.agent(
    adata,
    hidden_dim=256,      # More hidden units
    latent_dim=20,       # Higher latent dimension
    encoder_num_layers=3 # Deeper network
)

# Adjust regularization
model = iaode.agent(
    adata,
    beta=0.5,    # Lower KL weight if reconstruction is poor
    tc=0.5,      # Add total correlation for disentanglement
    dip=0.1      # Add DIP loss for interpretability
)
```

### 3. Training Strategy
```python
# Quick testing
model.fit(epochs=50, patience=10, val_every=5)

# Production training
model.fit(epochs=200, patience=30, val_every=5, early_stop=True)

# Check convergence
import matplotlib.pyplot as plt
plt.plot(model.train_losses, label='Train')
plt.plot(model.val_losses, label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

### 4. Memory Management
```python
# Reduce batch size if out of memory
model = iaode.agent(adata, batch_size=64)

# Or subsample data for initial testing
adata_sub = adata[:5000, :].copy()
model = iaode.agent(adata_sub)
```

## Next Steps

1. **Read the examples**: Check `examples/` for detailed use cases
2. **API Reference**: See README.md for complete API documentation
3. **Customize**: Experiment with different architectures and hyperparameters
4. **Evaluate**: Use built-in evaluation metrics to assess model quality
5. **Benchmark**: Compare against scVI and other methods

## Getting Help

- **Documentation**: Full README and examples
- **Issues**: GitHub Issues for bug reports
- **Questions**: Open a discussion on GitHub

## Citation

If you use iAODE in your research:

```bibtex
@software{iaode2024,
    author = {Zeyu Fu},
    title = {iAODE: Interpretable Autoencoder with Ordinary Differential Equations},
    year = {2024},
    url = {https://github.com/PeterPonyu/iAODE}
}
```
