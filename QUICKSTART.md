# Quick Start Guide - iAODE

For complete documentation, see `README.md`. This guide covers the most common patterns.

## Installation

```bash
pip install iaode
```

## 30-Second Example (scRNA-seq)

```python
import scanpy as sc
import iaode

adata = sc.datasets.paul15()  # Small scRNA-seq example
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers['counts'] = adata.X.copy()

model = iaode.agent(adata, layer='counts', latent_dim=10)
model.fit(epochs=50)

latent = model.get_latent()
adata.obsm['X_iaode'] = latent
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)
sc.pl.umap(adata, color='paul15_clusters')
```

## Common Patterns

### Custom Architecture

```python
model = iaode.agent(
    adata,
    latent_dim=10,
    hidden_dim=256,
    encoder_type='transformer',
    loss_mode='nb'
)
model.fit(epochs=100, patience=20)
```

### Trajectory Inference (Neural ODE)

```python
model = iaode.agent(adata, use_ode=True, latent_dim=10, i_dim=2)
model.fit(epochs=100)

pseudo_time = model.take_time(adata.X)
adata.obs['pseudo_time'] = pseudo_time
```

### scATAC-seq Peak Annotation

```python
adata = iaode.annotation_pipeline(
    h5_file='filtered_peak_bc_matrix.h5',
    gtf_file='gencode.v49.annotation.gtf.gz',
    promoter_upstream=2000,
    apply_tfidf=True,
    select_hvp=True
)
```

### Model Evaluation

```python
dr_metrics = iaode.evaluate_dimensionality_reduction(X_high=adata.X, X_low=latent, k=10)
ls_metrics = iaode.evaluate_single_cell_latent_space(latent_space=latent, data_type='trajectory')
```

## Tips

**Loss modes:** `mse` (continuous), `nb` (count data), `zinb` (zero-inflated counts)

**Training:** Quick test: `model.fit(epochs=50, patience=10)` | Production: `model.fit(epochs=200, patience=30, early_stop=True)`

**Memory:** Reduce batch size or subsample: `adata_sub = adata[:5000, :].copy()`

## Data Downloads

For scATAC examples, download from verified sources:
- GENCODE: [v19](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz) | [v49](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz)
- 10X: [5k PBMC](https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/) | [10k PBMC](https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_pbmc_10k_v2/)

## Next Steps

1. Check `examples/` for detailed use cases
2. See `README.md` for full API reference
3. Customize hyperparameters for your data
4. Evaluate model quality using built-in metrics

## Citation

```bibtex
@software{iaode2024,
    author = {Zeyu Fu},
    title = {iAODE: Interpretable Autoencoder with Ordinary Differential Equations},
    year = {2024},
    url = {https://github.com/PeterPonyu/iAODE}
}
```
