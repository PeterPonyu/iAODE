# iAODE: Interpretable Autoencoder with Ordinary Differential Equations

[![PyPI version](https://badge.fury.io/py/iaode.svg)](https://badge.fury.io/py/iaode)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for single-cell omics data analysis that combines variational autoencoders (VAE) with neural ordinary differential equations (ODE) for trajectory inference and dimensionality reduction.

## Features

- **Interpretable Dimensionality Reduction**: VAE-based architecture with multiple loss modes (MSE, NB, ZINB)
- **Trajectory Inference**: Neural ODE integration for continuous trajectory modeling
- **scATAC-seq Peak Annotation**: Comprehensive peak-to-gene annotation pipeline following best practices
- **Comprehensive Evaluation**: Built-in metrics for dimensionality reduction and latent space quality
- **Benchmark Framework**: Compare against state-of-the-art methods (scVI, PEAKVI, POISSONVI)
- **Flexible Architecture**: Support for multiple encoder types (MLP, Residual MLP, Linear, Transformer)

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

### Dependencies

- Python >= 3.9
- PyTorch >= 1.10.0
- AnnData >= 0.8.0
- Scanpy >= 1.8.0
- scvi-tools >= 0.16.0
- See `requirements.txt` for complete list

## Quick Start

> **📁 See `examples/` for complete runnable examples**

### Basic scRNA-seq Analysis

```python
import scanpy as sc
import iaode

# Load and preprocess
adata = sc.datasets.paul15()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

# Train iAODE model with key hyperparameters
model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=10,         # Latent space dimensions
    hidden_dim=128,        # Hidden layer size
    encoder_type='mlp',    # Options: 'mlp', 'residual_mlp', 'transformer', 'linear'
    loss_mode='nb',        # Options: 'mse', 'nb', 'zinb'
    use_ode=False          # Set True for trajectory inference
)

model.fit(epochs=100, patience=20, val_every=5)
latent = model.get_latent()

# Visualize with UMAP
adata.obsm['X_iaode'] = latent
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)
sc.pl.umap(adata, color='paul15_clusters')
```

### Trajectory Inference with Neural ODE

```python
# Enable Neural ODE for trajectory modeling
model = iaode.agent(
    adata,
    use_ode=True,          # Enable ODE
    i_dim=2,               # ODE intermediate dimension
    latent_dim=10,
    loss_mode='nb'
)

model.fit(epochs=100)
latent = model.get_latent()
iembed = model.get_iembed()  # ODE intermediate states

# Compute pseudotime from latent space
adata.obs['pseudotime'] = (latent[:, 0] - latent[:, 0].min()) / (latent[:, 0].max() - latent[:, 0].min())
```

### scATAC-seq Peak Annotation

iAODE provides a complete scATAC-seq preprocessing and annotation pipeline:

```python
import iaode

adata = iaode.annotation_pipeline(
    h5_file='filtered_peak_bc_matrix.h5',
    gtf_file='gencode.v49.annotation.gtf.gz',
    promoter_upstream=2000,
    apply_tfidf=True,
    select_hvp=True,
    n_top_peaks=20000
)
```

**Quick data download:** Use `examples/data/download_data.sh` to fetch reference files:

```bash
cd examples/data
./download_data.sh human 5k_pbmc
```

**Reference datasets:**

- GENCODE GTFs: [v19](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz) | [v49](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz) | [Mouse vM25](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz) | [Mouse vM38](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.annotation.gtf.gz)
- 10X samples: [5k PBMC](https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/) | [10k PBMC](https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_pbmc_10k_v2/) | [8k Cortex](https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_mouse_cortex_8k_v2/)

### Evaluation and Benchmarking

```python
from iaode import (
    evaluate_dimensionality_reduction,
    evaluate_single_cell_latent_space,
    DataSplitter,
    train_scvi_models
)

# Evaluate dimensionality reduction quality
dr_metrics = evaluate_dimensionality_reduction(
    X_high=adata.X,
    X_low=latent,
    k=10,
    verbose=True
)

# Evaluate latent space for single-cell data
ls_metrics = evaluate_single_cell_latent_space(
    latent_space=latent,
    data_type='trajectory',  # or 'steady_state'
    verbose=True
)

# Benchmark against scVI models
splitter = DataSplitter(
    n_samples=adata.n_obs,
    test_size=0.15,
    val_size=0.15,
    random_state=42
)

scvi_results = train_scvi_models(
    adata,
    splitter,
    n_latent=10,
    n_epochs=400,
    batch_size=128
)
```

## Model Architecture

### Core Components

1. **Encoder**: Maps input data to latent distribution
   - Types: MLP, Residual MLP, Linear, Transformer
   - Outputs: mean, log-variance, sampled latent vector

2. **Decoder**: Reconstructs data from latent representation
   - Supports MSE, NB, ZINB loss modes
   - Learns library size and dispersion parameters

3. **Neural ODE** (Optional): Models continuous trajectories
   - Latent ODE function for trajectory inference
   - Time encoder predicts pseudo-time

4. **Information Bottleneck**: Additional interpretable layer
   - Projects latent space to lower-dimensional embedding
   - Enhances interpretability

### Loss Components

- **Reconstruction**: MSE / NB / ZINB likelihood
- **KL Divergence**: Regularizes latent distribution
- **β-TCVAE**: Total correlation decomposition
- **DIP**: Disentanglement via learned projections  
- **MMD**: Maximum mean discrepancy
- **ODE Consistency**: Aligns VAE and ODE latents

## Examples

The `examples/` directory contains complete, runnable scripts demonstrating all major features. All outputs are automatically organized in `examples/outputs/<example_name>/` for easy access.

### Running the Examples

**Prerequisites**: Ensure iAODE is installed:

```bash
pip install iaode
# or from source:
cd /path/to/iAODE && pip install -e .
```

**Navigate to examples directory**:

```bash
cd examples
```

### Available Examples

#### 1. Basic Usage (`basic_usage.py`)
Demonstrates fundamental scRNA-seq dimensionality reduction with iAODE.

```bash
python basic_usage.py
```

**Features:**
- Installation verification and setup
- Paul15 dataset preprocessing
- Model training with inline hyperparameter documentation
- UMAP-based visualizations (cell types, latent dimensions)
- Outputs saved to `outputs/basic_usage/`

**Key hyperparameters explained:**
- `encoder_type='mlp'`: Encoder architecture (options: 'mlp', 'residual_mlp', 'transformer', 'linear')
- `loss_mode='nb'`: Loss function (options: 'mse', 'nb', 'zinb')
- `latent_dim=10`: Dimensionality of latent space
- `hidden_dim=128`: Hidden layer size

---

#### 2. Trajectory Inference (`trajectory_inference.py`)
Demonstrates Neural ODE-based trajectory modeling for developmental processes.

```bash
python trajectory_inference.py
```

**Features:**
- Neural ODE integration with `use_ode=True`
- Pseudotime computation and visualization
- Velocity field analysis (quiver + streamplot)
- UMAP-based trajectory visualization
- Outputs saved to `outputs/trajectory_inference/`

**Key configuration:**
- `use_ode=True`: Enable Neural ODE for continuous trajectories
- `i_dim=2`: Intermediate ODE state dimension for trajectory modeling

---

#### 3. Model Evaluation (`model_evaluation.py`)
Comprehensive benchmarking comparing iAODE against scVI-family models.

```bash
python model_evaluation.py
```

**Features:**
- Side-by-side comparison: iAODE, scVI, PEAKVI, POISSONVI
- **Consistent evaluation metrics** across all models:
  - **Dimensionality Reduction**: Distance Correlation, Q_local, Q_global
  - **Latent Space Quality**: Manifold Dimensionality, Spectral Decay, Trajectory Directionality
  - **Clustering**: NMI, ARI, ASW
- Comparison table saved as CSV
- Visual comparisons (bar plots + UMAP visualizations)
- Outputs saved to `outputs/model_evaluation/`

---

#### 4. scATAC-seq Annotation (`atacseq_annotation.py`)
Complete scATAC-seq peak annotation and preprocessing pipeline.

```bash
python atacseq_annotation.py
```

**Prerequisites**: Download required data files first:

```bash
cd data
bash download_data.sh
# This downloads:
# - mouse_brain_5k_v1.1.h5 (10X scATAC-seq data)
# - gencode.vM25.annotation.gtf (mouse gene annotations)
```

**Features:**
- Data availability checks with clear download instructions
- Peak-to-gene annotation (promoter/gene body/distal/intergenic)
- TF-IDF normalization
- Highly variable peak (HVP) selection
- Comprehensive QC visualizations (4-panel plot)
- Outputs saved to `outputs/atacseq_annotation/`

**QC visualizations:**
- Peak annotation type distribution
- Distance to TSS histogram
- Peak counts per cell
- Highly variable peak selection

---

### Output Organization

All examples save outputs to structured directories:

```
examples/
├── outputs/
│   ├── basic_usage/
│   │   ├── umap_celltypes.png
│   │   └── latent_dimensions.png
│   ├── trajectory_inference/
│   │   ├── trajectory_umap.png
│   │   └── velocity_field.png
│   ├── model_evaluation/
│   │   ├── model_comparison.png
│   │   └── model_comparison.csv
│   └── atacseq_annotation/
│       └── annotation_qc.png
```

### Customization

All examples can be adapted to your data:

1. **Load your AnnData object** instead of example datasets
2. **Adjust hyperparameters** based on your data characteristics:
   - Increase `latent_dim` for complex datasets (e.g., 20-50)
   - Use `encoder_type='transformer'` for large-scale data
   - Set `loss_mode='zinb'` for highly sparse data
3. **Modify visualization parameters** (colors, resolution, layout)

For detailed hyperparameter guidance, see inline comments in each example script.

## API Reference

### Main Classes

#### `iaode.agent`

High-level interface for model training and inference.

**Parameters:**
- `adata` (AnnData): Input single-cell data
- `layer` (str): Data layer to use (default: 'counts')
- `latent_dim` (int): Latent space dimension (default: 10)
- `hidden_dim` (int): Hidden layer dimension (default: 128)
- `use_ode` (bool): Enable neural ODE (default: False)
- `loss_mode` (str): Loss function ('mse', 'nb', 'zinb')
- `encoder_type` (str): Encoder architecture ('mlp', 'mlp_residual', 'linear', 'transformer')
- `lr` (float): Learning rate (default: 1e-4)
- `beta` (float): KL divergence weight (default: 1.0)
- `recon` (float): Reconstruction loss weight (default: 1.0)

**Methods:**
- `fit(epochs, patience, val_every, early_stop)`: Train model
- `get_latent()`: Get latent representation
- `get_iembed()`: Get interpretable embedding
- `get_test_latent()`: Get test set latent representation

### Annotation Functions

#### `iaode.annotation_pipeline`

Complete scATAC-seq peak annotation and preprocessing.

**Parameters:**
- `h5_file` (str): Path to 10X H5 file
- `gtf_file` (str): Path to GTF annotation
- `promoter_upstream` (int): TSS upstream extension (default: 2000)
- `promoter_downstream` (int): TSS downstream extension (default: 500)
- `apply_tfidf` (bool): Apply TF-IDF normalization (default: True)
- `select_hvp` (bool): Select highly variable peaks (default: True)
- `n_top_peaks` (int): Number of HVPs (default: 20000)

### Evaluation Functions

#### `iaode.evaluate_dimensionality_reduction`

Evaluate dimensionality reduction quality.

**Returns:** Dict with distance_correlation, Q_local, Q_global, K_max

#### `iaode.evaluate_single_cell_latent_space`

Evaluate single-cell latent space quality.

**Returns:** Dict with manifold_dimensionality, spectral_decay_rate, participation_ratio, etc.

## Advanced Usage

### Custom Encoder Architecture

```python
model = iaode.agent(
    adata,
    encoder_type='transformer',
    encoder_num_layers=4,
    encoder_n_heads=8,
    encoder_d_model=256,
    hidden_dim=512
)
```

### Multiple Loss Terms

```python
model = iaode.agent(
    adata,
    recon=1.0,      # Reconstruction loss
    beta=1.0,       # KL divergence
    tc=0.5,         # Total correlation
    dip=0.1,        # Disentanglement
    info=0.05       # MMD loss
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
        
    # Custom early stopping logic
    if should_stop(val_loss):
        model.load_best_model()
        break
```

## Citing iAODE

If you use iAODE in your research, please cite:

```bibtex
@software{iaode2025,
    author = {Zeyu Fu},
    title = {iAODE: Interpretable Autoencoder with Ordinary Differential Equations},
    year = {2025},
    url = {https://github.com/PeterPonyu/iAODE}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of PyTorch, AnnData, and Scanpy ecosystems
- Inspired by scVI-tools, Signac, and SnapATAC2 best practices
- Neural ODE implementation using torchdiffeq

## Contact

For questions and feedback:
- GitHub Issues: [https://github.com/PeterPonyu/iAODE/issues](https://github.com/PeterPonyu/iAODE/issues)
- Email: fuzeyu99@126.com

## Changelog

### v0.2.0 (2025-11-20)

**Documentation & Usability**
- Streamlined all documentation for clarity (README, QUICKSTART, examples/README)
- Separated scRNA-seq and scATAC-seq examples with clear workflows
- Added verified GENCODE and 10X Genomics reference data URLs
- Created `examples/data/download_data.sh` helper script for fetching reference files
- Removed Git LFS references in favor of external hosting guidance

**Code Quality**
- Cleaned example scripts: removed unused imports, simplified output messages
- Fixed Markdown formatting: proper code block spacing and language tags
- Ensured all API usage examples are correct and tested

**Package Improvements**
- Clarified that `paul15` dataset is scRNA-seq with standard preprocessing
- Highlighted complete scATAC-seq preprocessing + annotation pipeline
- Improved inline code documentation

### v0.1.2 (2025-11-19)

**Metadata & Contact**
- Updated author name to "Zeyu Fu" across all files
- Set primary contact email to fuzeyu99@126.com
- Fixed BibTeX citation formatting

### v0.1.1 (2025-11-19)

**Bug Fixes**
- Added missing `requests` dependency for scvi-tools compatibility
- Dropped Python 3.8 support (requires Python ≥3.9 for optax/scvi-tools)
- Fixed CI test imports and matrix configuration

### v0.1.0 (2025-11-19)

**Initial Release**
- VAE with neural ODE support for trajectory inference
- Complete scATAC-seq peak annotation pipeline
- Comprehensive evaluation metrics for dimensionality reduction
- Benchmark framework for comparing against scVI models
- Support for multiple encoder types (MLP, Residual MLP, Linear, Transformer)
- Multiple loss modes (MSE, NB, ZINB) for different data types
