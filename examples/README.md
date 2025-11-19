# iAODE Examples

This directory contains example scripts demonstrating various use cases of the iAODE package.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)
Demonstrates the basic workflow for training iAODE on scRNA-seq data.

**Topics covered:**
- Loading and preprocessing data
- Creating and training a basic VAE model
- Extracting latent representations
- Visualizing results with UMAP

**Run:**
```bash
python basic_usage.py
```

### 2. Trajectory Inference (`trajectory_inference.py`)
Shows how to use neural ODE for modeling continuous cellular trajectories.

**Topics covered:**
- Enabling neural ODE in the model
- Learning pseudo-time
- Extracting interpretable embeddings
- Computing cell-cell transitions
- Visualizing trajectory velocity fields

**Run:**
```bash
python trajectory_inference.py
```

### 3. scATAC-seq Peak Annotation (`atacseq_annotation.py`)
Complete pipeline for annotating scATAC-seq peaks to genes.

**Topics covered:**
- Loading 10X scATAC-seq data
- Peak-to-gene annotation using GTF files
- TF-IDF normalization
- Highly variable peak selection
- Quality control visualization

**Requirements:**
- 10X scATAC-seq H5 file
- Gene annotation GTF file (GENCODE/Ensembl)

**Run:**
```bash
python atacseq_annotation.py
```

### 4. Model Evaluation (`model_evaluation.py`)
Comprehensive evaluation and benchmarking against other methods.

**Topics covered:**
- Evaluating dimensionality reduction quality
- Assessing latent space properties
- Benchmarking against scVI models
- Comparing multiple methods
- Generating comparison visualizations

**Run:**
```bash
python model_evaluation.py
```

## Prerequisites

Install iAODE with all dependencies:
```bash
pip install iaode
```

For development:
```bash
pip install iaode[dev]
```

## Data Requirements

Most examples use publicly available datasets from Scanpy:
- `sc.datasets.paul15()`: Mouse hematopoiesis data

For the scATAC-seq example, you'll need:
- Your own 10X scATAC-seq data or download from 10X Genomics
- Gene annotation file from GENCODE or Ensembl

## Output

Each example generates:
- Console output with training progress and metrics
- PNG figures for visualization
- (Optional) CSV files with results

## Customization

All examples can be easily adapted to your own data:
1. Replace the dataset loading code
2. Adjust hyperparameters as needed
3. Modify visualization settings

## Tips

- Start with `basic_usage.py` to understand the core workflow
- Use GPU for faster training (automatically detected if available)
- Adjust `batch_size` based on your available memory
- Experiment with different `encoder_type` options for your data

## Support

For issues or questions:
- GitHub Issues: https://github.com/PeterPonyu/iAODE/issues
- Documentation: https://iaode.readthedocs.io
