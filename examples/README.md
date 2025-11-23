# iAODE Examples

This folder contains end‑to‑end, **reproducible examples** that demonstrate how to use `iaode` for:

- Trajectory inference with Neural ODE (scRNA‑seq & scATAC‑seq)
- scATAC‑seq peak annotation and QC
- Model evaluation and benchmarking against scVI‑family models

All examples are designed to:

- Be **stand‑alone** (just run the script)
- Automatically **download and cache** public datasets
- Produce **publication‑quality figures** and **summary statistics**
- Use a shared helper module `_example_utils.py` for logging and output management

---

## 1. Prerequisites

### Python environment

- Python 3.8+ (recommended)
- `iaode` and its dependencies installed
- Access to the internet for automatic dataset download on first run

A minimal setup might look like:

```bash
pip install iaode scanpy matplotlib scipy pandas
```

> The examples call `check_iaode_installed()` and will exit cleanly if `iaode` is not available.

### Data download & caching

All examples that use external data:

- Download and cache datasets under:  
  `~/.iaode/data/`
- Re‑use the cached files on subsequent runs (no repeated downloads).

You do **not** need to download any data manually.

---

## 2. Running the examples

From the `examples/` directory:

```bash
python <example_script>.py
```

Each script:

- Adds the example directory to `sys.path`
- Imports helper functions from `_example_utils.py`
- Sets up an example‑specific output directory via `setup_output_dir(<name>)`

Outputs (figures, CSVs, `.h5ad` files) are written into a dedicated folder, announced at the start of each run, for example:

- `examples_output/trajectory_inference/`
- `examples_output/trajectory_inference_atac/`
- `examples_output/atacseq_annotation/`
- `examples_output/model_evaluation/`

---

## 3. Shared utilities (`_example_utils.py`)

All examples use a small helper module that provides:

- `check_iaode_installed()` – verify `iaode` is importable
- `setup_output_dir(name)` – create/run‑specific output folder
- Pretty printing helpers:
  - `print_header()`
  - `print_section()`
  - `print_info()`
  - `print_success()`
  - `print_warning()`

These functions give a **consistent, readable CLI log** across all examples.

---

## 4. Example overview

Below is a summary of each example script based on its actual content.

### 4.1 Basic scATAC‑seq usage – dimensionality reduction

**Script:** `basic_usage.py`

**Goal:**  
Introduce basic iAODE usage with scATAC‑seq data: peak annotation, TF‑IDF normalization, model training, and UMAP visualization of the learned latent space.

**Dataset:**

- 10X Mouse Brain 5k scATAC‑seq

**Key steps:**

1. **Load and annotate scATAC‑seq data**
   - Automatic download:

     ```python
     h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
     ```

   - Annotation pipeline:

     ```python
     adata = iaode.annotation_pipeline(
         h5_file=str(h5_file),
         gtf_file=str(gtf_file),
         promoter_upstream=2000,
         promoter_downstream=500,
         apply_tfidf=True,
         select_hvp=True,
         n_top_peaks=20000,
     )
     ```

2. **Train iAODE**
   - Configuration:
     - `latent_dim=32`
     - `hidden_dim=512`
     - `encoder_type='mlp'`
     - `loss_mode='nb'` (negative binomial for scATAC‑seq counts)
     - `batch_size=128`
   - Fit with early stopping:

     ```python
     model.fit(epochs=100, patience=20, val_every=5)
     ```

   - Record resource metrics (train time, epochs, peak GPU memory).

3. **Extract latent representations**
   - `latent = model.get_latent()`
   - Store in `adata.obsm['X_iaode']`
   - Compute UMAP on latent space.

4. **2×2 visualization figure**

   Panels include:

   - **A.** UMAP colored by latent dimension 1 (viridis colormap)
   - **B.** UMAP colored by latent dimension 2 (plasma colormap)
   - **C.** UMAP colored by total peak counts (YlOrRd colormap)
   - **D.** UMAP colored by per‑cell latent variance (coolwarm colormap)

   Saved as:

   - `latent_space_analysis.png`
   - `latent_space_analysis.pdf`

5. **Latent distribution analysis**

   Additional 1×3 figure showing histograms of the first three latent dimensions with mean lines.

   Saved as:

   - `latent_distributions.png`
   - `latent_distributions.pdf`

6. **Summary statistics**

   - Dataset: cells × peaks
   - Training: time, epochs, peak GPU memory
   - Latent space: mean ± std, range for each dimension
   - Peak counts per cell

---

### 4.2 Trajectory inference with Neural ODE – scRNA‑seq

**Script:** `trajectory_inference_rna.py`

**Goal:**  
Demonstrate trajectory inference on the hematopoietic `paul15` dataset using `iaode` with a **Neural ODE**, including a computed **velocity field** and rich visualization.

**Key steps:**

1. **Load and preprocess data**
   - `adata = sc.datasets.paul15()`
   - Filter cells: `min_genes = 200`
   - Store raw counts in `adata.layers['counts']`
   - Normalize and log‑transform for Scanpy visualization (not for the loss):

     ```python
     sc.pp.normalize_total(adata, target_sum=1e4)
     sc.pp.log1p(adata)
     ```

   - Train iAODE with Neural ODE
   - Configuration:
     - `use_ode=True`
     - `i_dim=2` (interpretable ODE bottleneck)
     - `latent_dim=32`
     - `hidden_dim=512`
     - `loss_mode='nb'` (negative binomial on counts)
   - Fit with early stopping:

     ```python
     model.fit(epochs=50, patience=20, val_every=5)
     ```

   - Resource metrics via `model.get_resource_metrics()` (train time, epochs, peak GPU memory).

3. **Extract trajectory representations**
   - `latent = model.get_latent()`
   - `iembed = model.get_iembed()` (2D interpretable ODE state)
   - `pseudotime = model.get_pseudotime()`
   - Store in `adata.obsm` / `adata.obs`
   - Compute UMAP on `X_iaode` latent space.

4. **Velocity field**
   - Estimate velocity in latent and UMAP spaces:

     ```python
     E_grid, V_grid = model.get_vfres(
         adata,
         zs_key='X_iaode',
         E_key='X_umap',
         vf_key='X_vf_latent',
         dv_key='X_vf_umap',
         stream=True,
         density=1.5,
         smooth=0.5,
         n_neigh=n_neighbors,
         run_neigh=False,
     )
     ```

   - Compute velocity magnitude and store in `adata.obs['velocity_magnitude']`.

5. **Publication‑quality 3×3 figure**

   Panels include:

   - **A.** UMAP colored by normalized pseudotime
   - **B.** UMAP colored by `paul15_clusters` (cell type annotations), with a compact legend
   - **C.** UMAP colored by velocity magnitude
   - **D.** Streamplot of the velocity field overlaid on cells
   - **E.** Quiver plot (subsampled arrows) showing local velocity directions
   - **F.** Histogram of normalized pseudotime (mean/median lines)
   - **G.** Histogram of velocity magnitude (mean/median lines)
   - **H.** I‑embed 2D bottleneck space colored by pseudotime

   Saved as:

   - `trajectory_analysis.png`
   - `trajectory_analysis.pdf`

6. **Summary statistics**

   - Dataset: number of cells and genes
   - Training: time, time per epoch, epochs, peak GPU memory
   - Pseudotime: range, mean ± std, median
   - Velocity (latent & UMAP): mean ± std, max
   - Correlation between normalized pseudotime and velocity magnitude:
     - Pearson \(r\) and Spearman \(\rho\) with p‑values

---

### 4.3 Trajectory inference with Neural ODE – scATAC‑seq

**Script:** `trajectory_inference_atac.py`

**Goal:**  
Apply Neural ODE trajectory inference to **chromatin accessibility** data and visualize dynamics in UMAP space and an interpretable ODE bottleneck.

**Dataset:**

- 10X Mouse Brain 5k scATAC‑seq (subset of highly variable peaks)

**Key steps:**

1. **Load & annotate scATAC‑seq**
   - Automatic download:

     ```python
     h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
     ```

   - Annotation pipeline:

     ```python
     adata = iaode.annotation_pipeline(
         h5_file=str(h5_file),
         gtf_file=str(gtf_file),
         promoter_upstream=2000,
         promoter_downstream=500,
         apply_tfidf=True,
         select_hvp=True,
         n_top_peaks=20000,
     )
     ```

   - Subset to `adata.var['highly_variable']` if present.
   - Ensure `adata.layers['counts']` holds (TF‑IDF) normalized counts.
   - Compute per‑cell peak accessibility summary (`adata.obs['n_peaks']`).

2. **Train iAODE with Neural ODE**
   - Configuration:
     - `use_ode=True`
     - `i_dim=2`
     - `latent_dim=32`
     - `hidden_dim=512`
     - `loss_mode='nb'` (appropriate for scATAC count data)
   - Fit with early stopping.
   - Resource metrics recorded.

3. **Extract trajectory representations & UMAP**
   - Same workflow as the scRNA‑seq example: `get_latent()`, `get_iembed()`, `get_pseudotime()`
   - UMAP on `X_iaode`
   - Normalize pseudotime to \([0,1]\).

4. **Velocity field**
   - Same pattern as in scRNA‑seq, but applied to the scATAC latent space and UMAP.

5. **3×3 multi‑panel figure**

   Panels:

   - **A.** Pseudotime on UMAP (chromatin trajectory)
   - **B.** Peak accessibility (per‑cell peak counts)
   - **C.** Velocity magnitude
   - **D.** Velocity streamplot on UMAP
   - **E.** UMAP quiver plot
   - **F.** Pseudotime distribution
   - **G.** Velocity magnitude distribution
   - **H.** I‑embed 2D space colored by pseudotime

   Saved as:

   - `trajectory_analysis.png`
   - `trajectory_analysis.pdf`

6. **Summary**

   - Cells × peaks (HVP subset)
   - Accessibility statistics
   - Training metrics
   - Pseudotime statistics
   - Latent/UMAP velocity statistics
   - Pseudotime–velocity correlation
   - Peak annotation distribution (if `annotation_type` is available)

---

### 4.4 scATAC‑seq peak annotation & QC

**Script:** `atacseq_annotation.py`

**Goal:**  
Provide a **complete peak annotation pipeline** for scATAC‑seq:

- Map peaks to genes using GTF annotations
- Characterize genomic context (promoter, gene body, distal, intergenic)
- Generate QC visualizations and save an annotated `.h5ad`

**Dataset:**

- 10X Mouse Brain 5k scATAC‑seq  
- GENCODE vM25 GTF annotation

**Key steps:**

1. **Download data**
   - `iaode.datasets.mouse_brain_5k_atacseq()`  
     Returns paths to the 10X H5 file and GTF.
   - Files cached in `~/.iaode/data/`.

2. **Run annotation pipeline**

   ```python
   adata = iaode.annotation_pipeline(
       h5_file=str(h5_file),
       gtf_file=str(gtf_file),
       promoter_upstream=2000,
       promoter_downstream=500,
       apply_tfidf=True,
       select_hvp=True,
       n_top_peaks=20000,
   )
   ```

   - Annotates peaks to genes and genomic features
   - Performs TF‑IDF normalization
   - Selects highly variable peaks (HVPs)

3. **Compute quick statistics**
   - Per‑cell peak counts (coverage)
   - If available: distance to TSS (`adata.var['distance_to_tss']`), summarized (mean/median).

4. **2×2 QC figure**

   Uses a **colorblind‑friendly palette** and consistent styling.

   Panels:

   - **A. Peak annotation distribution (pie chart)**  
     - e.g. promoter, gene body, distal, intergenic  
     - Customized colors for each annotation type
   - **B. Distance to TSS distribution**  
     - Histogram of distances (±50 kb, plotted in kb)  
     - Mean and median distance lines
   - **C. Peak counts per cell**  
     - Histogram of per‑cell peak counts with mean/median
   - **D. Highly variable peaks selection**  
     - Bar plot of HVP vs. other peaks with counts and percentages

   Saved as:

   - `annotation_qc.png`
   - `annotation_qc.pdf`

5. **Save annotated data**

   - Annotated object written as:

     ```python
     adata.write_h5ad("annotated_peaks.h5ad")
     ```

6. **Detailed textual summary**

   - Dataset overview (cells, peaks)
   - Genomic annotation distribution (ASCII bar visualization)
   - HVP statistics
   - Cell quality metrics (mean/median/range of peaks per cell)
   - TSS distance statistics (if available)
   - Suggested **next steps**:
     - Use `iaode.agent()` on the annotated data
     - Perform trajectory inference with `use_ode=True`
     - Link peaks to genes for gene activity scoring

---

### 4.5 Model evaluation & benchmarking – scRNA‑seq (LSE on paul15)

**Script:** `model_evaluation_rna.py`

**Goal:**  
Benchmark `iAODE` against **scVI‑family models** on the `paul15` trajectory dataset using **Latent Space Evaluation (LSE)** metrics:

- Manifold dimensionality
- Spectral decay rate
- Trajectory directionality

**Key steps:**

1. **Load and preprocess `paul15`**
   - As in the trajectory example:
     - Filter cells
     - Save raw counts to `adata.layers['counts']`
     - Normalize & log transform for downstream use

2. **Train/val/test split**

   Uses `iaode.DataSplitter`:

   - `test_size = 0.15`
   - `val_size = 0.15`
   - `random_state = 42`

3. **Train iAODE**

   - Neural ODE model with:
     - `latent_dim=32`
     - `hidden_dim=512`
     - `loss_mode='nb'`
   - Early stopping:
     - `epochs=100`, `patience=20`, `val_every=5`
   - Store test‑set latent representation and resource metrics.

4. **Compute LSE metrics for iAODE**

   On the **test latent space**:

   ```python
   ls_metrics = iaode.evaluate_single_cell_latent_space(
       latent_space=latent_iaode_test,
       data_type='trajectory',
       verbose=True,
   )
   ```

5. **Train and evaluate scVI‑family models**

   - Uses a convenience function:

     ```python
     scvi_results = iaode.train_scvi_models(
         adata, splitter,
         n_latent=CONFIG['latent_dim'],
         n_epochs=CONFIG['epochs'],
         batch_size=CONFIG['batch_size'],
     )
     ```

   - For each available model (e.g. `scvi`, `scanvi`, `peakvi`, `poissonvi`):
     - Extract test latent representations
     - Evaluate with the same LSE function
     - Catch and report any failures per model

6. **Comparison table**

   - Build a `pandas.DataFrame` summarizing:

     | Model | Train Time (s) | Epochs | Manifold Dim | Spectral Decay | Trajectory Dir |
     |-------|----------------|--------|--------------|----------------|----------------|

   - Printed to stdout and saved as:
     - `model_comparison.csv`

7. **UMAP for visualization**

   - For each model:
     - Store its latent representation in `adata_viz.obsm['X_latent']`
     - Run neighbors + UMAP
     - Store in `results[model_name]['adata_viz']`

8. **Publication‑quality 2×4 figure**

   - **Row 1: Metrics**
     - **A.** Manifold dimensionality (bar plot)
     - **B.** Spectral decay rate (bar plot)
     - **C.** Trajectory directionality (bar plot)
     - **D.** Training time (bar plot)
   - **Row 2: UMAP projections**
     - Up to four models (e.g. iAODE, scVI variants) visualized as:
       - UMAP colored by `paul15_clusters`
       - Shared categorical legend across the bottom
   - Uses a colorblind‑friendly palette for models and cluster colors.

   Saved as:

   - `model_comparison.png`
   - `model_comparison.pdf`

9. **Summary printout**

   - Configuration (epochs, latent dimension)
   - Number of cells and test set size
   - Per‑model summary with:

     \[
     \text{Train time},\ \text{ManifoldDim},\ \text{SpectralDecay},\ \text{TrajDir}
     \]

   - List of output files.

---

### 4.6 Model evaluation & benchmarking – scATAC‑seq

**Script:** `model_evaluation_atac.py`

**Goal:**  
Benchmark `iAODE` against **scVI‑family models** on scATAC‑seq data using **Latent Space Evaluation (LSE)** metrics, with comprehensive UMAP visualizations across models.

**Dataset:**

- 10X Mouse Brain 5k scATAC‑seq (HVP subset)

**Key steps:**

1. **Load and annotate scATAC‑seq data**
   - Download and annotate data using `iaode.annotation_pipeline()`
   - Subset to highly variable peaks (HVPs) for computational efficiency
   - Configuration: `n_top_peaks=20000`

2. **Train/val/test split**

   Uses `iaode.DataSplitter`:

   - `test_size = 0.15`
   - `val_size = 0.15`
   - `random_state = 42`

3. **Train iAODE**

   - Neural ODE model with:
     - `latent_dim=32`
     - `hidden_dim=512`
     - `use_ode=True`
     - `loss_mode='nb'` (negative binomial for scATAC‑seq)
   - Early stopping:
     - `epochs=100`, `patience=20`, `val_every=5`
   - Store test‑set latent representation and resource metrics.

4. **Compute LSE metrics for iAODE**

   On the **test latent space**:

   ```python
   ls_metrics = iaode.evaluate_single_cell_latent_space(
       latent_space=latent_iaode_test,
       data_type='trajectory',
       verbose=True,
   )
   ```

5. **Train and evaluate scVI‑family models**

   - Uses a convenience function:

     ```python
     scvi_results = iaode.train_scvi_models(
         adata, splitter,
         n_latent=CONFIG['latent_dim'],
         n_epochs=CONFIG['epochs'],
         batch_size=CONFIG['batch_size'],
     )
     ```

   - For each available model (e.g. `peakvi`, `poissonvi`, `scvi`, `scanvi`):
     - Extract test latent representations
     - Evaluate with the same LSE function
     - Catch and report any failures per model

6. **Comparison table**

   - Build a `pandas.DataFrame` summarizing:

     | Model | Train Time (s) | Epochs | Manifold Dim | Spectral Decay | Trajectory Dir |
     |-------|----------------|--------|--------------|----------------|----------------|

   - Printed to stdout and saved as:
     - `model_comparison.csv`

7. **UMAP for visualization**

   - For each model:
     - Store its latent representation in `adata_viz.obsm['X_latent']`
     - Run neighbors + UMAP
     - Store in `results[model_name]['adata_viz']`

8. **Publication‑quality 3×n figure**

   Multi‑panel layout with up to 5 models:

   - **Row 1: Metrics**
     - **A.** Manifold dimensionality (bar plot)
     - **B.** Spectral decay rate (bar plot)
     - **C.** Trajectory directionality (bar plot)
     - **D.** Training time (bar plot)
   - **Row 2: UMAP colored by peak counts**
     - One panel per model (e.g. E–I)
     - Shared colorbar and axis limits
   - **Row 3: UMAP colored by latent dimension 1**
     - One panel per model (e.g. J–N)
     - Shared colorbar and axis limits
   
   Uses a colorblind‑friendly palette for models.

   Saved as:

   - `model_comparison.png`
   - `model_comparison.pdf`

9. **Summary printout**

   - Configuration (epochs, latent dimension, HVP count)
   - Number of cells and test set size
   - Per‑model summary with:

     \[
     \text{Train time},\ \text{ManifoldDim},\ \text{SpectralDecay},\ \text{TrajDir}
     \]

   - List of output files.

---

## 5. Tips for extending these examples

- Start from the example **closest to your data type**:
  - scRNA‑seq trajectory → `trajectory_inference_rna.py` (Neural ODE paul15 example)
  - scATAC‑seq basic usage → `basic_usage.py` (dimensionality reduction)
  - scATAC‑seq workflow → `atacseq_annotation.py` + `trajectory_inference_atac.py` (Annotation + Neural ODE)
  - Method benchmarking → `model_evaluation_rna.py` or `model_evaluation_atac.py` (LSE benchmarking)
- Replace the dataset loading step with your own `AnnData` object.
- Keep the `layers['counts']` convention for raw counts.
- Re‑use:
  - The plotting helpers (UMAP styling, colorbar utilities)
  - The evaluation pattern with `evaluate_single_cell_latent_space`
  - The `DataSplitter` approach for robust benchmarking.

---

## 6. Outputs at a glance

Each example writes:

- **Figures**: `.png` and `.pdf` under an example‑specific subdirectory
- **Tables / CSVs**: e.g. `model_comparison.csv`
- **Annotated data**: e.g. `annotated_peaks.h5ad` for the scATAC annotation example

All output paths are printed at the end of the run so you can quickly locate the generated artifacts.

---
