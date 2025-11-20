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

### Basic Usage

```python
import anndata as ad
import iaode

# Load your single-cell data
adata = ad.read_h5ad('your_data.h5ad')

# Create and train model
model = iaode.agent(
    adata,
    layer='counts',
    latent_dim=10,
    hidden_dim=128,
    use_ode=True,          # Enable neural ODE for trajectory
    loss_mode='nb',        # Negative binomial loss for count data
    encoder_type='mlp'     # MLP encoder architecture
)

# Train with early stopping
model.fit(
    epochs=100,
    patience=20,
    val_every=5,
    early_stop=True
)

# Get latent representation
latent = model.get_latent()

# Get interpretable embedding
iembed = model.get_iembed()
```

### scATAC-seq Peak Annotation

```python
import iaode

# Complete annotation pipeline
adata = iaode.annotation_pipeline(
    h5_file='filtered_peak_bc_matrix.h5',
    gtf_file='gencode.v44.annotation.gtf',
    output_h5ad='annotated_peaks.h5ad',
    
    # Annotation parameters
    promoter_upstream=2000,
    promoter_downstream=500,
    gene_body=True,
    gene_type='protein_coding',
    
    # TF-IDF normalization
    apply_tfidf=True,
    tfidf_scale_factor=1e4,
    
    # Highly variable peaks
    select_hvp=True,
    n_top_peaks=20000,
    hvp_method='signac'
)
```

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
@software{iaode2024,
    author = {Zeyu Fu},
    title = {iAODE: Interpretable Autoencoder with Ordinary Differential Equations},
    year = {2024},
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

### v0.1.0 (2024-11-19)
- Initial release
- VAE with neural ODE support
- scATAC-seq peak annotation pipeline
- Comprehensive evaluation metrics
- Benchmark framework for scVI models
