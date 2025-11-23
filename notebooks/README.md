# iAODE Jupyter Notebooks

This directory contains Jupyter notebook versions of the iAODE examples, converted from the Python scripts in the `examples/` directory.

## 📓 Available Notebooks

### Basic Examples

1. **[01_basic_usage.ipynb](01_basic_usage.ipynb)**
   - Basic scATAC-seq dimensionality reduction
   - Dataset: 10X Mouse Brain 5k
   - Features: TF-IDF normalization, peak annotation, UMAP visualization

2. **[02_atacseq_annotation.ipynb](02_atacseq_annotation.ipynb)**
   - Complete peak-to-gene annotation pipeline
   - Genomic feature annotation, QC visualizations
   - Highly variable peak (HVP) selection

### Trajectory Inference

3. **[03_trajectory_inference_rna.ipynb](03_trajectory_inference_rna.ipynb)**
   - Neural ODE trajectory inference on scRNA-seq
   - Dataset: paul15 (hematopoietic differentiation)
   - Features: pseudotime, velocity fields, interpretable bottleneck

4. **[04_trajectory_inference_atac.ipynb](04_trajectory_inference_atac.ipynb)**
   - Neural ODE trajectory inference on scATAC-seq
   - Dataset: 10X Mouse Brain 5k
   - Features: chromatin accessibility dynamics, velocity visualization

### Model Evaluation & Benchmarking

5. **[05_model_evaluation_rna.ipynb](05_model_evaluation_rna.ipynb)**
   - Benchmark iAODE vs scVI-family models (scRNA-seq)
   - Latent Space Evaluation (LSE) metrics
   - Dataset: paul15 trajectory

6. **[06_model_evaluation_atac.ipynb](06_model_evaluation_atac.ipynb)**
   - Benchmark iAODE vs scVI-family models (scATAC-seq)
   - Dataset: 10X Mouse Brain 5k HVP subset
   - Comprehensive model comparison

## 🚀 Getting Started

### Prerequisites

Install iAODE and dependencies:

```bash
pip install iaode
# Or for development:
pip install -e ".[dev]"
```

### Running Notebooks

1. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   Or use VS Code with the Jupyter extension.

2. **Open a notebook** and run cells sequentially (Shift+Enter).

3. **Outputs** are saved to `outputs/<notebook_name>/` by default.

### Environment Variables

Customize output locations:
- `IAODE_OUTPUT_ROOT`: Set custom output directory (default: `outputs/`)
- `IAODE_OUTPUT_FLAT`: Set to `1` for flat output structure

Example:
```bash
export IAODE_OUTPUT_ROOT=/path/to/outputs
jupyter notebook
```

## 📊 Expected Outputs

Each notebook generates:
- **Publication-quality figures** (PNG + PDF)
- **Statistical summaries** and metrics
- **Annotated AnnData objects** (where applicable)

### Example Output Structure

```
outputs/
├── basic_usage/
│   ├── latent_space_analysis.png
│   ├── latent_space_analysis.pdf
│   └── latent_distributions.{png,pdf}
├── atacseq_annotation/
│   ├── annotation_qc.{png,pdf}
│   └── annotated_peaks.h5ad
├── trajectory_inference/
│   └── trajectory_analysis.{png,pdf}
└── model_evaluation/
    └── model_comparison.csv
```

## 💡 Tips

- **Run cells sequentially** – notebooks are designed to be executed in order
- **Check outputs** – figures are saved automatically, check the `outputs/` directory
- **GPU acceleration** – if available, PyTorch will automatically use GPU
- **Adjust epochs** – reduce `epochs` parameter for faster testing
- **Memory management** – start with smaller datasets if you encounter memory issues

## 🔗 Related Resources

- **Python scripts:** `../examples/` directory
- **Documentation:** [README.md](../README.md)
- **Examples guide:** [examples/README.md](../examples/README.md)

## 📝 Notes

- **Conversion:** These notebooks were automatically generated from Python scripts
- **Functionality:** All original features and visualizations are preserved
- **Testing:** You can test these notebooks and provide feedback
- **Modifications:** Feel free to modify parameters and experiment

## 🐛 Troubleshooting

### Data Download Issues
Data is automatically downloaded to `~/.iaode/data/`. If downloads fail:
```python
import iaode
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
```

### Import Errors
Ensure iaode is installed:
```bash
pip install --upgrade iaode
```

### Memory Issues
Reduce batch size or subset data:
```python
model = iaode.agent(adata, batch_size=64)  # smaller batch size
adata = adata[:1000, :].copy()  # subset to 1000 cells
```

## 📬 Feedback

Found an issue or have suggestions? Please open an issue on [GitHub](https://github.com/PeterPonyu/iAODE/issues).
