# iAODE Datasets Guide

## Overview

iAODE provides automatic dataset downloading and caching functionality, similar to popular packages like Scanpy. This eliminates the need for manual file management and ensures reproducible analyses.

## Quick Start

```python
import iaode

# Automatically download and cache mouse brain scATAC-seq data
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()

# Run annotation pipeline
adata = iaode.annotation_pipeline(
    h5_file=str(h5_file),
    gtf_file=str(gtf_file),
    apply_tfidf=True,
    select_hvp=True
)
```

## Available Datasets

### Mouse Brain 5k scATAC-seq

**Function:** `iaode.datasets.mouse_brain_5k_atacseq()`

- **Dataset:** 10X Genomics Mouse Brain scATAC-seq (5,337 cells, 157,797 peaks)
- **Size:** ~920 MB total
  - H5 matrix: 73 MB
  - GENCODE vM25 annotation: 847 MB (decompressed)
- **Files cached in:** `~/.iaode/data/`
- **First run:** Downloads both files automatically
- **Subsequent runs:** Uses cached versions

**Usage:**

```python
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
```

### Human PBMC 5k scATAC-seq

**Function:** `iaode.datasets.human_pbmc_5k_atacseq()`

- **Dataset:** 10X Genomics PBMC scATAC-seq (5,155 cells, ~100k peaks)
- **Size:** ~1.5 GB total
  - H5 matrix: ~50 MB
  - GENCODE v49 annotation: 1.4 GB (decompressed)
- **Files cached in:** `~/.iaode/data/`

**Usage:**

```python
h5_file, gtf_file = iaode.datasets.human_pbmc_5k_atacseq()
```

## Cache Management

### View Cached Files

```python
import iaode

iaode.datasets.list_cached_files()
```

**Output:**
```
======================================================================
iAODE Cached Datasets
======================================================================
Location: /home/user/.iaode/data

  atacseq/mouse_brain_5k_v1.1.h5
    Size: 72.5 MB
  annotations/gencode.vM25.annotation.gtf
    Size: 847.3 MB

Total: 2 files, 919.8 MB
======================================================================
```

### Clear Cache

Free up disk space by removing all cached datasets:

```python
iaode.datasets.clear_cache()
```

Files will be automatically re-downloaded on next use.

## Smart Local File Detection

If you've already downloaded files manually (e.g., to `examples/data/`), iAODE will automatically detect and use them:

```python
# Run from examples/ directory
# Checks: ./data/, ../data/, ~/.iaode/data/
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
```

**Search order:**
1. `examples/data/` (if running from examples directory)
2. `~/.iaode/data/` (user cache)
3. Downloads from web if not found

## Handling Download Failures

Some 10X Genomics URLs may have access restrictions. If automatic download fails:

```python
# Error message provides manual download instructions
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
```

**Output:**
```
⚠️  Automatic download failed: HTTP Error 403: Forbidden

======================================================================
MANUAL DOWNLOAD REQUIRED
======================================================================
Please download the file manually:

1. Visit: https://www.10xgenomics.com/datasets/...
   Or try direct link: https://cf.10xgenomics.com/...

2. Save as: /home/user/.iaode/data/atacseq/mouse_brain_5k_v1.1.h5

3. Or place in: /path/to/examples/data/mouse_brain_5k_v1.1.h5
======================================================================
```

## Using Custom Datasets

You can use your own scATAC-seq data with the same pipeline:

```python
import iaode

# Your custom H5 and GTF files
adata = iaode.annotation_pipeline(
    h5_file="path/to/your_data.h5",
    gtf_file="path/to/annotation.gtf",
    promoter_upstream=2000,
    promoter_downstream=500,
    apply_tfidf=True,
    select_hvp=True,
    n_top_peaks=20000
)
```

## Storage Locations

### Default Cache Directory

```
~/.iaode/data/
├── atacseq/
│   ├── mouse_brain_5k_v1.1.h5
│   └── human_pbmc_5k_nextgem.h5
└── annotations/
    ├── gencode.vM25.annotation.gtf  (mouse)
    └── gencode.v49.annotation.gtf   (human)
```

### Alternative Locations

iAODE also checks these locations (in order):

1. `$PWD/data/` (current working directory)
2. `$PWD/../data/` (parent directory, useful when running from `examples/`)
3. `~/.iaode/data/` (user cache)

## Best Practices

### For Examples and Tutorials

```python
# Use automatic download - simplest for users
import iaode

h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
adata = iaode.annotation_pipeline(h5_file, gtf_file)
```

### For Production Pipelines

```python
# Use your own data with explicit paths
import iaode

adata = iaode.annotation_pipeline(
    h5_file="/data/project/sample_001.h5",
    gtf_file="/reference/gencode.vM38.gtf",
    promoter_upstream=3000,  # Custom parameters
    apply_tfidf=True,
    select_hvp=True,
    n_top_peaks=30000
)
```

### For Reproducible Research

```python
# Document exact dataset versions
import iaode

# Mouse GENCODE vM25 (release date: 2020-05)
# 10X Mouse Brain 5k v1.1 (chemistry: Next GEM)
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()

# Check cache location for version tracking
iaode.datasets.list_cached_files()
```

## Troubleshooting

### Issue: Download fails with 403 Forbidden

**Solution:** Download manually from 10X Genomics website and place in:
- `~/.iaode/data/atacseq/`
- Or `examples/data/` if running examples

### Issue: Out of disk space

**Solution:** Clear cache to free up ~2 GB:

```python
iaode.datasets.clear_cache()
```

### Issue: GTF decompression is slow

**Reason:** GENCODE GTF files are large (1-1.5 GB decompressed)

**Solution:** First download takes time (~2-3 minutes). Subsequent runs use cached version instantly.

### Issue: Wrong annotation version

**Solution:** Force re-download latest version:

```python
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq(force_download=True)
```

## Comparison with Other Packages

### Scanpy-style automatic downloads

```python
# Scanpy
import scanpy as sc
adata = sc.datasets.paul15()

# iAODE (similar API)
import iaode
h5, gtf = iaode.datasets.mouse_brain_5k_atacseq()
```

### Advantages

- **Automatic caching:** No duplicate downloads
- **Smart detection:** Uses local files if available
- **Progress tracking:** Shows download progress with MB/s
- **Graceful failure:** Clear manual download instructions
- **Version consistency:** Same files across all users

## API Reference

### `iaode.datasets.mouse_brain_5k_atacseq(force_download=False, use_local=True)`

Download and cache mouse brain scATAC-seq dataset.

**Parameters:**
- `force_download` (bool): Re-download even if cached (default: False)
- `use_local` (bool): Check local directories before downloading (default: True)

**Returns:**
- `h5_path` (Path): Path to H5 peak matrix
- `gtf_path` (Path): Path to GTF annotation

### `iaode.datasets.human_pbmc_5k_atacseq(force_download=False)`

Download and cache human PBMC scATAC-seq dataset.

**Parameters:**
- `force_download` (bool): Re-download even if cached (default: False)

**Returns:**
- `h5_path` (Path): Path to H5 peak matrix
- `gtf_path` (Path): Path to GTF annotation

### `iaode.datasets.list_cached_files()`

Display all cached datasets and their sizes.

**Returns:** None (prints to console)

### `iaode.datasets.clear_cache()`

Remove all cached datasets to free disk space.

**Returns:** None

### `iaode.datasets.get_data_dir()`

Get the iAODE data cache directory path.

**Returns:** `Path` object pointing to `~/.iaode/data/`

## Examples

### Complete scATAC-seq Pipeline

```python
import iaode
import scanpy as sc

# 1. Download data
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()

# 2. Annotate peaks
adata = iaode.annotation_pipeline(
    h5_file=str(h5_file),
    gtf_file=str(gtf_file),
    promoter_upstream=2000,
    apply_tfidf=True,
    select_hvp=True,
    n_top_peaks=20000
)

# 3. Train model
model = iaode.agent(
    adata,
    layer='X_tfidf',
    latent_dim=20,
    use_ode=False
)
model.fit(epochs=100)

# 4. Visualize
latent = model.get_latent()
adata.obsm['X_iaode'] = latent
sc.pp.neighbors(adata, use_rep='X_iaode')
sc.tl.umap(adata)
sc.pl.umap(adata, color='n_peaks')
```

### Benchmarking Across Species

```python
import iaode

# Mouse dataset
mouse_h5, mouse_gtf = iaode.datasets.mouse_brain_5k_atacseq()
mouse_adata = iaode.annotation_pipeline(mouse_h5, mouse_gtf)

# Human dataset
human_h5, human_gtf = iaode.datasets.human_pbmc_5k_atacseq()
human_adata = iaode.annotation_pipeline(human_h5, human_gtf)

# Compare peak annotation patterns
print(f"Mouse annotated: {mouse_adata.var['annotation'].value_counts()}")
print(f"Human annotated: {human_adata.var['annotation'].value_counts()}")
```
