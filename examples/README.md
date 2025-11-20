# iAODE Examples

Comprehensive examples demonstrating **iAODE** (interpretable Accessibility ODE VAE) for single-cell chromatin accessibility analysis and trajectory inference.

---

## 🎯 **What is iAODE?**

**iAODE** = **I**nterpretable **A**ccessibility **O**rdinary **D**ifferential **E**quation VAE

A deep generative model for scATAC-seq data that combines:
- **Variational Autoencoder (VAE)**: Dimensionality reduction with probabilistic reconstruction (NB/ZINB)
- **Interpretable Bottleneck**: Explicit `i_dim` factors (`X_iembed`) capturing regulatory modules
- **Neural ODE**: Optional trajectory dynamics for pseudotime and velocity inference
- **Multi-modal Support**: Works with both scATAC-seq (peaks) and scRNA-seq (genes)

---

## 📁 **Example Files Overview**

### 🔥 **Recommended for scATAC-seq**

| Script | Purpose | Data Source | Key Outputs |
|--------|---------|-------------|-------------|
| **`scATAC_quickstart.py`** | Complete scATAC pipeline (TF-IDF → HVP → iAODE) | User H5 + GTF | Latent, iembed, pseudotime, velocity |
| **`atacseq_annotation.py`** | Peak-to-gene annotation with QC plots | Auto-downloads Mouse Brain 5k | Annotated peaks + visualizations |
| **`trajectory_ode.py`** | Neural ODE trajectory on synthetic data | Synthetic sparse | Pseudotime + velocity extraction |

### 📊 **Evaluation & Benchmarking**

| Script | Purpose | Data Source | Key Metrics |
|--------|---------|-------------|-------------|
| **`evaluation_metrics.py`** | Dimensionality reduction + clustering quality | Synthetic data | Distance corr, Q_local/global, ARI, NMI |
| **`model_evaluation.py`** | Comprehensive benchmarking vs scVI models | User data | Train/val/test metrics, DRE + LSE |

### 🧬 **Legacy scRNA-seq Examples**

| Script | Purpose | Data Source | Notes |
|--------|---------|-------------|-------|
| **`basic_usage.py`** | Simple scRNA-seq workflow | paul15 (auto) | Basic training + UMAP |
| **`trajectory_inference.py`** | scRNA trajectory with vector field | paul15 (auto) | Uses `get_vfres()` for streamplot |

---

## 🚀 **Quick Start: scATAC-seq**

### **1. End-to-End Pipeline** (`scATAC_quickstart.py`)

Process your own 10X scATAC-seq data:

```bash
python scATAC_quickstart.py \
    --h5 filtered_peak_bc_matrix.h5 \
    --gtf gencode.vM25.annotation.gtf.gz \
    --use-ode \
    --latent-dim 32 \
    --hidden-dim 512 \
    --i-dim 16 \
    --loss-mode zinb \
    --epochs 400
```

**Fast dry run** (subset 5000 cells):
```bash
python scATAC_quickstart.py --h5 filtered_peak_bc_matrix.h5 --subsample 5000
```

**Outputs**:
- `iaode_scATAC_processed.h5ad` containing:
  - `obsm['X_iaode']` — Latent representation (z)
  - `obsm['X_iembed']` — **Interpretable factors** (i_dim bottleneck)
  - `obs['pseudotime']` — ODE time parameter (if `--use-ode`)
  - `obsm['velocity']` — Latent velocity field (if `--use-ode`)

---

### **2. Peak Annotation Pipeline** (`atacseq_annotation.py`)

Automatically download demo data and annotate peaks:

```bash
python atacseq_annotation.py
```

**What it does**:
1. Downloads Mouse Brain 5k scATAC-seq + GENCODE vM25 GTF (cached in `~/.iaode/data/`)
2. Annotates peaks → promoter/exonic/intronic/intergenic
3. Computes distance to TSS
4. TF-IDF normalization
5. Selects highly variable peaks (HVPs)
6. Generates QC plots

**Outputs**:
- `results/annotated_peaks.h5ad`
- `outputs/atacseq_annotation/annotation_qc.png`

---

### **3. Neural ODE Trajectory** (`trajectory_ode.py`)

Minimal example with synthetic data (no files needed):

```bash
python trajectory_ode.py --latent-dim 24 --i-dim 12 --epochs 300
```

**Outputs**:
- `trajectory_ode_output.h5ad` with pseudotime and velocity
- Demonstrates ODE-based trajectory inference

---

## 📊 **Evaluation Examples**

### **1. Dimensionality Reduction Quality** (`evaluation_metrics.py`)

```bash
python evaluation_metrics.py
```

**Metrics computed**:
- **Distance correlation**: Global structure preservation (Spearman ρ)
- **Q_local / Q_global**: Local vs global neighborhood quality
- **Clustering metrics**: ARI, NMI, ASW, Calinski-Harabasz, Davies-Bouldin

```

---

### **2. Model Benchmarking** (`model_evaluation.py`)

Compare iAODE against scVI family models:

```bash
python model_evaluation.py
```

**Models compared**:
- iAODE (this work)
- scVI (Negative Binomial VAE)
- PEAKVI (scATAC-specific VAE)
- POISSONVI (Poisson VAE)

**Evaluation framework**:
- **DRE** (Dimensionality Reduction Evaluator): Distance correlation, Q metrics
- **LSE** (Latent Space Evaluator): Manifold consistency, spectral decay, participation ratio, anisotropy
- **Clustering**: ARI, NMI, silhouette score
- **Resource metrics**: Training time, GPU memory

---

## 🧬 **Legacy scRNA-seq Examples**

### **1. Basic Usage** (`basic_usage.py`)
```bash
python basic_usage.py
```
- paul15 hematopoietic differentiation dataset
- Simple training + UMAP visualization

### **2. Trajectory Inference** (`trajectory_inference.py`)
```bash
python trajectory_inference.py
```
- paul15 with Neural ODE
- Velocity field visualization using `model.get_vfres()`

---

## 🔬 **Data Requirements**

### **For Real scATAC-seq Analysis**

**Minimum**:
- 10X Genomics filtered peak matrix: `filtered_peak_bc_matrix.h5`

**Recommended** (for annotation):
- Matching GTF annotation file

**Suggested datasets**:
| Species | Dataset | GTF Version | Size |
|---------|---------|-------------|------|
| Mouse | 10X Mouse Brain 5k | GENCODE vM25 | ~5k cells, ~100k peaks |
| Human | 10X PBMC 5k | GENCODE v49 | ~5k cells, ~120k peaks |

### **For Demo/Testing**
- `atacseq_annotation.py` auto-downloads Mouse Brain 5k
- Other examples use synthetic or auto-downloaded data (paul15)

---

## ⚙️ **Preprocessing Pipeline**

Standard workflow (implemented in examples):

```python
# 1. Load data
adata = load_10x_h5_data('filtered_peak_bc_matrix.h5')
adata.layers['counts'] = adata.X.copy()

# 2. TF-IDF normalization (Signac/SnapATAC2 best practice)
tfidf_normalization(adata, scale_factor=1e4, log_tf=False, log_idf=True)

# 3. Highly variable peak (HVP) selection
select_highly_variable_peaks(
    adata, 
    n_top_peaks=20000,
    method='signac',  # or 'snapatac2', 'deviance'
    min_accessibility=0.01,
    max_accessibility=0.95
)

# 4. Train iAODE
model = iaode.agent(
    adata, 
    layer='counts',
    latent_dim=32,
    hidden_dim=512,
    i_dim=16,
    use_ode=True,
    loss_mode='zinb'
)
model.fit(epochs=400, patience=25, val_every=10)

# 5. Extract representations
latent = model.get_latent()          # Latent space (z)
iembed = model.get_iembed()          # Interpretable factors
pseudotime = model.get_pseudotime()  # ODE time (if use_ode=True)
velocity = model.get_velocity()      # Latent velocity (if use_ode=True)
```

---

## 🎨 **Interpretability: Understanding `X_iembed`**

The **interpretable bottleneck** (`i_dim`) captures regulatory modules:

```python
model = iaode.agent(adata, latent_dim=32, i_dim=16, ...)
iembed = model.get_iembed()  # Shape: (n_cells, i_dim=16)
```

**What are interpretable factors?**
- Explicit low-dimensional bottleneck before latent space
- Each factor represents a regulatory module or cell state axis
- Can be correlated with TFs, chromatin states, or biological pathways

**Visualization**:
```python
import scanpy as sc
adata.obsm['X_iembed'] = iembed

# UMAP on interpretable factors
sc.pp.neighbors(adata, use_rep='X_iembed')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['Factor_1', 'Factor_2'])
```

---

## 🧭 **Trajectory & Velocity Inference**

### **Neural ODE Dynamics**

When `use_ode=True`, iAODE learns latent ODE dynamics:

```
dz/dt = f_ode(z, t)
```

**Key methods**:
- `model.get_pseudotime()` → ODE time parameter (t)
- `model.get_velocity()` → Latent velocity field (dz/dt)
- `model.get_vfres()` → Vector field for streamplot visualization

**Example**:
```python
model = iaode.agent(adata, use_ode=True, i_dim=8, ...)
model.fit(epochs=400)

# Extract trajectory information
pseudotime = model.get_pseudotime()
velocity = model.get_velocity()

# Visualize velocity field (requires UMAP)
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)

E_grid, V_grid = model.get_vfres(
    adata,
    zs_key='X_iaode',
    E_key='X_umap',
    stream=True,
    density=1.5
)
```

---

## 🔧 **Customization Guide**

### **Architecture Parameters**

| Parameter | Recommendation | Reasoning |
|-----------|---------------|-----------|
| `latent_dim` | 10-50 | 32 for typical scATAC, 50+ for large atlas |
| `i_dim` | latent_dim / 2 | Interpretable bottleneck (8-16 typical) |
| `hidden_dim` | 256-512 | 512 for deep regulatory complexity |
| `encoder_type` | `'mlp'` or `'transformer'` | Transformer for >20k cells + many peaks |
| `loss_mode` | `'zinb'` | Best for ultra-sparse scATAC (use `'nb'` for scRNA) |

### **Training Parameters**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epochs` | 400 | Increase to 600 for large datasets |
| `batch_size` | 128-256 | Reduce if OOM, increase for speed |
| `patience` | 20-30 | Early stopping patience |
| `val_every` | 5-10 | Validation frequency |

### **ODE Parameters**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_ode` | `False` | Set `True` for trajectory inference |
| `i_dim` | Required | Intermediate ODE state dimension |
| `beta_kl` | Auto-tuned | KL divergence weight (adjust if collapse) |

---

## 📚 **Evaluation Metrics Reference**

### **Dimensionality Reduction (DRE)**
- **Distance Correlation**: Spearman ρ between high-dim and low-dim pairwise distances (higher = better global structure)
- **Q_local**: Local neighborhood preservation quality (higher = better)
- **Q_global**: Global structure preservation quality (higher = better)

### **Latent Space Quality (LSE)**
- **Manifold Dimensionality Consistency**: How efficiently variance is captured (0-1, higher = better)
- **Spectral Decay Rate**: Eigenvalue concentration (higher = better for trajectories)
- **Participation Ratio**: Dimensional balance (lower = better for trajectories, higher for steady-state)
- **Anisotropy Score**: Directionality strength (higher = better for trajectories)

### **Clustering**
- **ARI** (Adjusted Rand Index): Overlap with ground truth (0-1, higher = better)
- **NMI** (Normalized Mutual Info): Information agreement (0-1, higher = better)
- **ASW** (Silhouette Score): Cluster separation (-1 to 1, higher = better)
- **Calinski-Harabasz**: Cluster compactness (higher = better)
- **Davies-Bouldin**: Cluster separation (lower = better)

---

## 🤝 **Contributing**

Found issues or want to add examples? Open an issue or PR:
- **GitHub**: https://github.com/PeterPonyu/iAODE
- **Issues**: https://github.com/PeterPonyu/iAODE/issues

---

## 📄 **License**

MIT License - See LICENSE file for details.

---

## 🔗 **Related Resources**

- **Documentation**: [Link to main docs when available]
- **Paper**: [Link to preprint/publication]
- **Data Sources**:
  - 10X Genomics: https://www.10xgenomics.com/datasets
  - GENCODE: https://www.gencodegenes.org/

---

**Last Updated**: 2025 | **iAODE Version**: 1.0.0