# iAODE: Interpretable Accessibility ODE VAE for scATAC-seq

[![PyPI version](https://badge.fury.io/py/iaode.svg)](https://badge.fury.io/py/iaode) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`iAODE` (interpretable Accessibility ODE VAE) is a lightweight deep learning framework purpose-built for single‑cell ATAC‑seq (scATAC‑seq) data. It couples a variational autoencoder (VAE) with a Neural ODE and an interpretable bottleneck to jointly achieve:

1. **Robust modeling** of sparse count accessibility profiles (NB / ZINB likelihoods)
2. **Continuous trajectory inference** (Neural ODE pseudotime + velocity)
3. **Interpretable latent factors** (biologically aligned bottleneck)
4. **Scalable preprocessing** (TF‑IDF + highly variable peak selection) aligned with Signac / SnapATAC2 best practices

The design targets the unique characteristics of scATAC data: extreme sparsity, library size variability, heterogeneous peak accessibility kinetics, and dynamic regulatory trajectories.

---

## Core Architecture

```
Raw Peak Counts  --(TF-IDF)-->  Normalized Matrix  --(HVP Selection)-->  Peak Subset
       |
       v
Encoder (MLP / Residual / Transformer) --> q(z|x)
       |                 \
       |                  +--> Bottleneck (i_dim) -> Interpretable factors
       v
   Neural ODE f(z,t) -> z_ode(t)  (pseudotime + dynamics)
       |                \
       |                 +--> Consistency Loss (q_z vs z_ode)
       v
   Decoder (NB / ZINB / MSE) -> Reconstruction x_hat
```

**Key Components**:
- **TF‑IDF Normalization**: Stabilizes cell‑wise peak depth, emphasizes specific accessibility
- **Highly Variable Peaks (HVP)**: Variance / VMR / deviance‑based selection
- **NB/ZINB Likelihoods**: Models over‑dispersion & zero inflation
- **Neural ODE**: Smooth accessibility progression with pseudotime & velocity
- **Interpretable Bottleneck**: Linear compression preserving biologically decodable axes
- **Multi‑Objective Regularization**: β‑VAE KL, β‑TC, DIP, InfoVAE MMD, ODE consistency

**Loss Function**:
```
Loss = recon + i_recon + ODE_consistency + β·KL + dip·DIP + tc·TC + info·MMD
```

---

## Installation

### From PyPI (Recommended)

```bash
pip install iaode
```

### From Source

```bash
git clone https://github.com/PeterPonyu/iAODE.git
cd iAODE
pip install -e .
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.10.0
- AnnData ≥ 0.8.0
- Scanpy ≥ 1.8.0
- scvi-tools ≥ 0.16.0

See `requirements.txt` for complete dependencies.

---

## Quick Start

### Basic scATAC-seq Workflow

```python
import scanpy as sc
import iaode
from iaode.utils import tfidf_normalization, select_highly_variable_peaks
from iaode.annotation import load_10x_h5_data

# 1. Download and load scATAC-seq data (auto-cached)
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
print(f"Data downloaded to: {h5_file.parent}")

# Load peak count matrix
adata = load_10x_h5_data(str(h5_file))
adata.layers['counts'] = adata.X.copy()
print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} peaks")

# 2. TF-IDF normalization (Signac/SnapATAC2 best practice)
tfidf_normalization(
    adata,
    scale_factor=1e4,
    log_tf=False,
    log_idf=True,
    inplace=True
)

# 3. Highly variable peak (HVP) selection
select_highly_variable_peaks(
    adata,
    n_top_peaks=20000,
    method='signac',
    min_accessibility=0.01,
    max_accessibility=0.95,
    inplace=True
)

# Subset to HVPs
hvp_mask = adata.var['highly_variable']
adata = adata[:, hvp_mask].copy()
print(f"Retained {adata.n_vars} highly variable peaks")

# 4. Train iAODE model
model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=32,         # Higher for scATAC complexity
    hidden_dim=512,        # Deeper for regulatory patterns
    encoder_type='mlp',
    loss_mode='zinb',      # Best for sparse scATAC data
    use_ode=False
)

model.fit(epochs=400, patience=25, val_every=10)

# 5. Extract latent representation
latent = model.get_latent()
adata.obsm['X_iaode'] = latent

# 6. Visualize with UMAP
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)

# Color by QC metrics
sc.pl.umap(adata, color=['n_genes_by_counts', 'total_counts'])
```

### scATAC-seq with Trajectory Inference

```python
import scanpy as sc
import iaode
from iaode.utils import tfidf_normalization, select_highly_variable_peaks
from iaode.annotation import load_10x_h5_data

# 1. Load and preprocess scATAC-seq data
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
adata = load_10x_h5_data(str(h5_file))
adata.layers['counts'] = adata.X.copy()

# TF-IDF normalization
tfidf_normalization(adata, scale_factor=1e4, log_tf=False, log_idf=True, inplace=True)

# Select highly variable peaks
select_highly_variable_peaks(adata, n_top_peaks=20000, method='signac', inplace=True)
adata = adata[:, adata.var['highly_variable']].copy()

# 2. Train iAODE with Neural ODE for trajectory inference
model = iaode.agent(
    adata,
    layer='counts',
    use_ode=True,          # Enable Neural ODE dynamics
    i_dim=16,              # Interpretable bottleneck dimension
    latent_dim=32,
    hidden_dim=512,
    encoder_type='mlp',
    loss_mode='zinb'
)

model.fit(epochs=400, patience=25, val_every=10)

# 3. Extract trajectory-related representations
latent = model.get_latent()           # Latent space (z)
iembed = model.get_iembed()           # Interpretable regulatory factors
pseudotime = model.get_pseudotime()   # ODE time parameter
velocity = model.get_velocity()       # Latent velocity field

# Store in AnnData
adata.obsm['X_iaode'] = latent
adata.obsm['X_iembed'] = iembed
adata.obs['pseudotime'] = pseudotime
adata.obsm['velocity'] = velocity

# 4. Visualize trajectory
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)

# Color UMAP by pseudotime to reveal developmental trajectory
sc.pl.umap(adata, color='pseudotime', cmap='viridis')

# Visualize velocity field (requires UMAP coordinates)
E_grid, V_grid = model.get_vfres(
    adata,
    zs_key='X_iaode',
    E_key='X_umap',
    stream=True,
    density=1.5
)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata, color='pseudotime', ax=ax, show=False)
ax.streamplot(E_grid[0], E_grid[1], V_grid[0], V_grid[1], 
              color='gray', density=1.5, linewidth=0.5, arrowsize=1)
plt.tight_layout()
plt.show()
```

### scATAC-seq Peak Annotation Pipeline

**Automatic data download** - iAODE automatically downloads and caches datasets:

```python
import iaode

# Downloads mouse brain 5k scATAC-seq + GENCODE vM25 GTF
# Files cached in ~/.iaode/data/ and reused on subsequent runs
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()

# Run complete annotation pipeline
adata = iaode.annotation_pipeline(
    h5_file=str(h5_file),
    gtf_file=str(gtf_file),
    promoter_upstream=2000,  # TSS upstream region
    promoter_downstream=500,  # TSS downstream region
    apply_tfidf=True,        # TF-IDF normalization
    select_hvp=True,         # Select highly variable peaks
    n_top_peaks=20000        # Number of HVPs to retain
)

# AnnData now contains:
# - adata.var['peak_type']: promoter/exonic/intronic/intergenic
# - adata.var['gene_name']: Associated gene names
# - adata.var['distance_to_tss']: Distance to nearest TSS
# - adata.var['highly_variable']: HVP selection mask
# - Preprocessed with TF-IDF and HVP selection
```

**Available datasets:**

```python
# Mouse brain 5k scATAC-seq
h5, gtf = iaode.datasets.mouse_brain_5k_atacseq()

# Human PBMC 5k scATAC-seq
h5, gtf = iaode.datasets.human_pbmc_5k_atacseq()

# Cache management
iaode.datasets.list_cached_files()  # Show cached files
iaode.datasets.clear_cache()        # Clear all cached data
```


---

## Examples

Comprehensive examples are available in the `examples/` directory. See **[examples/README.md](examples/README.md)** for detailed documentation.

### Example Scripts

| Script | Purpose | Modality | Best For |
|--------|---------|----------|----------|
| **`basic_usage.py`** | scATAC-seq with 2D/histogram visualizations | scATAC | Getting started with scATAC-seq |
| **`atacseq_annotation.py`** | Peak-to-gene annotation + QC plots | scATAC | Understanding peak annotation |
| **`model_evaluation_atac.py`** | Benchmark iAODE vs scVI on scATAC-seq | scATAC | Comparative analysis (chromatin) |
| **`model_evaluation_rna.py`** | Benchmark iAODE vs scVI on scRNA-seq | scRNA | Comparative analysis (transcriptome) |
| **`trajectory_inference_atac.py`** | Neural ODE trajectory with scATAC-seq | scATAC | Chromatin accessibility dynamics |
| **`trajectory_inference_rna.py`** | Neural ODE trajectory with scRNA-seq | scRNA | Transcriptional trajectory |


### Running Examples

```bash
cd examples

# Getting started with scATAC-seq
python basic_usage.py

# Peak-to-gene annotation with auto-download
python atacseq_annotation.py

# Benchmark models on scATAC-seq data
python model_evaluation_atac.py

# Trajectory inference with Neural ODE (scRNA-seq)
python trajectory_inference_rna.py
```

All examples save outputs to `examples/outputs/<example_name>/` by default. See **[examples/README.md](examples/README.md)** for detailed usage instructions and customization options.

---

## API Reference

### `iaode.agent` - Main Model Interface

**Initialization:**

```python
model = iaode.agent(
    adata,                    # AnnData object
    layer='counts',           # Data layer to use
    latent_dim=10,            # Latent space dimension
    hidden_dim=128,           # Hidden layer dimension
    i_dim=None,               # Interpretable bottleneck dim (required if use_ode=True)
    use_ode=False,            # Enable Neural ODE
    loss_mode='nb',           # Loss function: 'mse', 'nb', 'zinb'
    encoder_type='mlp',       # Encoder: 'mlp', 'residual_mlp', 'transformer', 'linear'
    lr=1e-4,                  # Learning rate
    batch_size=128,           # Batch size
    beta=1.0,                 # KL divergence weight
    recon=1.0,                # Reconstruction loss weight
    tc=0.0,                   # Total correlation weight
    dip=0.0,                  # DIP weight
    info=0.0                  # InfoVAE MMD weight
)
```

**Training:**

```python
model.fit(
    epochs=100,               # Maximum epochs
    patience=20,              # Early stopping patience
    val_every=5,              # Validation frequency
    early_stop=True           # Enable early stopping
)
```

**Representation Extraction:**

```python
# Basic representations
latent = model.get_latent()              # Latent space (n_cells, latent_dim)
iembed = model.get_iembed()              # Interpretable factors (n_cells, i_dim)

# Trajectory-specific (requires use_ode=True)
pseudotime = model.get_pseudotime()      # ODE time parameter (n_cells,)
velocity = model.get_velocity()          # Latent velocity (n_cells, latent_dim)

# Vector field for visualization (requires UMAP in adata.obsm['X_umap'])
E_grid, V_grid = model.get_vfres(
    adata,
    zs_key='X_iaode',        # Latent representation key
    E_key='X_umap',          # Embedding key for visualization
    stream=True,             # Return streamplot-compatible format
    density=1.5              # Grid density
)
```

**Evaluation Metrics:**

```python
# Training metrics
metrics = model.get_resource_metrics()
# Returns: {'train_time': float, 'actual_epochs': int, 'peak_memory_gb': float}
```

---

### `iaode.annotation_pipeline` - scATAC-seq Preprocessing

**Complete annotation and preprocessing pipeline:**

```python
adata = iaode.annotation_pipeline(
    h5_file,                    # Path to 10X H5 file
    gtf_file,                   # Path to GTF annotation
    promoter_upstream=2000,     # TSS upstream extension (bp)
    promoter_downstream=500,    # TSS downstream extension (bp)
    apply_tfidf=True,           # Apply TF-IDF normalization
    select_hvp=True,            # Select highly variable peaks
    n_top_peaks=20000,          # Number of HVPs to retain
    hvp_method='signac',        # HVP method: 'signac', 'snapatac2', 'deviance'
    min_accessibility=0.01,     # Min peak accessibility fraction
    max_accessibility=0.95      # Max peak accessibility fraction
)
```

**Returns AnnData with:**
- `adata.var['peak_type']`: Peak annotation (promoter/exonic/intronic/intergenic)
- `adata.var['gene_name']`: Associated gene names
- `adata.var['distance_to_tss']`: Distance to nearest TSS
- `adata.var['highly_variable']`: HVP selection mask
- Preprocessed and normalized counts

---

### Evaluation Functions

#### Dimensionality Reduction Quality

```python
from iaode import evaluate_dimensionality_reduction

metrics = evaluate_dimensionality_reduction(
    X_high,                  # High-dimensional data (n_cells, n_features)
    X_low,                   # Low-dimensional embedding (n_cells, n_latent)
    k=10,                    # Number of neighbors
    verbose=True
)

# Returns:
# - distance_correlation: Global structure preservation (Spearman ρ)
# - Q_local: Local neighborhood quality
# - Q_global: Global structure quality
# - K_max: Local-global transition point
```

#### Latent Space Quality

```python
from iaode import evaluate_single_cell_latent_space

metrics = evaluate_single_cell_latent_space(
    latent_space,            # Latent representation (n_cells, n_latent)
    data_type='trajectory',  # 'trajectory' or 'steady_state'
    verbose=True
)

# Returns:
# - manifold_dimensionality: Dimensional efficiency (0-1)
# - spectral_decay_rate: Eigenvalue concentration
# - participation_ratio: Dimensional balance
# - anisotropy_score: Directionality strength
# - trajectory_directionality: Dominant axis strength
# - noise_resilience: Signal-to-noise ratio
# - overall_quality: Aggregate score
```

#### Model Benchmarking

```python
from iaode import DataSplitter, train_scvi_models, evaluate_scvi_models

# Create consistent train/val/test splits
splitter = DataSplitter(
    n_samples=adata.n_obs,
    test_size=0.15,
    val_size=0.15,
    random_state=42
)

# Train scVI family models
scvi_results = train_scvi_models(
    adata,
    splitter,
    n_latent=10,
    n_epochs=400,
    batch_size=128
)

# Evaluate all models
scvi_metrics = evaluate_scvi_models(
    scvi_results,
    adata,
    splitter.test_idx
)
```

---

### Preprocessing Utilities

```python
from iaode.utils import tfidf_normalization, select_highly_variable_peaks

# TF-IDF normalization (Signac/SnapATAC2 style)
tfidf_normalization(
    adata,
    scale_factor=1e4,
    log_tf=False,
    log_idf=True,
    inplace=True
)

# Highly variable peak selection
select_highly_variable_peaks(
    adata,
    n_top_peaks=20000,
    method='signac',         # or 'snapatac2', 'deviance'
    min_accessibility=0.01,
    max_accessibility=0.95,
    inplace=True
)
```

---

## Advanced Usage

### Custom Encoder Architecture

```python
# Transformer encoder for large-scale data
model = iaode.agent(
    adata,
    encoder_type='transformer',
    encoder_num_layers=4,
    encoder_n_heads=8,
    encoder_d_model=256,
    hidden_dim=512,
    latent_dim=32
)
```

### Multi-Objective Regularization

```python
# Fine-tune regularization weights
model = iaode.agent(
    adata,
    recon=1.0,      # Reconstruction loss
    beta=1.0,       # KL divergence
    tc=0.5,         # Total correlation (disentanglement)
    dip=0.1,        # DIP (dimension-wise independence)
    info=0.05       # InfoVAE MMD (distribution matching)
)
```

### Custom Training Loop

```python
# Manual training with custom logic
for epoch in range(100):
    train_loss = model.train_epoch()
    
    if epoch % 5 == 0:
        val_loss, val_score = model.validate()
        print(f"Epoch {epoch}: Val Loss={val_loss:.4f}")
        
    if should_stop(val_loss):
        model.load_best_model()
        break
```

---

## Citation

If you use iAODE in your research, please cite:

```bibtex
@software{iaode2025,
    author = {Zeyu Fu},
    title = {iAODE: Interpretable Accessibility ODE VAE for Single-Cell Chromatin Dynamics},
    year = {2025},
    publisher = {GitHub},
    url = {https://github.com/PeterPonyu/iAODE}
}
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed changes.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built upon ideas from:
- **scVI-tools**: scVI, PEAKVI, POISSONVI architectures
- **Signac** and **SnapATAC2**: scATAC-seq best practices
- **Neural ODE** literature: Continuous latent dynamics

PyTorch, AnnData, and Scanpy ecosystems provide the foundation.

---

## Contact

- **GitHub Issues**: [https://github.com/PeterPonyu/iAODE/issues](https://github.com/PeterPonyu/iAODE/issues)
- **Email**: fuzeyu99@126.com

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes and version history.
