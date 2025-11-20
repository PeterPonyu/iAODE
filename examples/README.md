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

**Requirements:**

**Run:**
```bash
python atacseq_annotation.py
```

### 4. Model Evaluation (`model_evaluation.py`)
**Topics covered:**
- Evaluating dimensionality reduction quality
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
- 10X scATAC-seq H5 file
- GENCODE/Ensembl GTF annotation

Place both files in `examples/data/` before running the example. See `README.md` for verified data download links, or use the provided download helper:

```bash
cd examples/data
./download_data.sh human 5k_pbmc  # Download human GTF and 5k PBMC sample
```

## Verified Data URLs

Below are verified, commonly used reference files and example datasets you can download and place into `examples/data/`.

GENCODE annotations
- Human v19 (GRCh37/hg19): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz
- Mouse vM25 (GRCm38/mm10): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz
- Human v49 (GRCh38/hg38): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz
- Mouse vM38 (GRCm39/mm39): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.annotation.gtf.gz

10X Genomics scATAC-seq example datasets (base URLs)
- 5k Human PBMCs (ATAC v1.1): https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/
- 10k Human PBMCs (ATAC v2): https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_pbmc_10k_v2/
- 8k Mouse Cortex (ATAC v2): https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_mouse_cortex_8k_v2/

Quick download examples:

```bash
# Download a GENCODE GTF (example: human v49)
wget -P examples/data/ https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz

# Example: download 10X filtered peak matrix (open the base URL and pick the appropriate file name, e.g. 'filtered_peak_bc_matrix.h5')
wget -P examples/data/ https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/filtered_peak_bc_matrix.h5
```

Note: 10X sample directories may contain tarballs or multiple files; open the base URL in a browser to identify the correct file if the direct name is not available.

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
