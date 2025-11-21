"""
Dataset utilities for iAODE examples.

Provides automatic download and caching of example datasets,
including scATAC-seq data and reference annotations.
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path
from typing import Optional, Tuple
import anndata as ad  # type: ignore


def get_data_dir() -> Path:
    """
    Get the iAODE data cache directory.
    
    Creates directory structure:
    ~/.iaode/data/
    ├── atacseq/
    └── annotations/
    
    Returns
    -------
    Path
        Path to data cache directory
    """
    data_dir = Path.home() / ".iaode" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "atacseq").mkdir(exist_ok=True)
    (data_dir / "annotations").mkdir(exist_ok=True)
    return data_dir


def _download_file(url: str, output_path: Path, desc: str = "file") -> None:
    """
    Download a file with progress indication.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path
        Local path to save file
    desc : str
        Description for progress messages
    """
    print(f"📥 Downloading {desc}...")
    print(f"   URL: {url}")
    print(f"   → {output_path}")
    
    try:
        # Create a custom progress hook
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(blocknum * blocksize / totalsize * 100, 100)
                mb_downloaded = blocknum * blocksize / 1024 / 1024
                mb_total = totalsize / 1024 / 1024
                print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
        
        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        print()  # New line after progress
        print(f"   ✓ Downloaded successfully")
    except Exception as e:
        print(f"\n   ✗ Download failed: {e}")
        raise


def _decompress_gz(gz_path: Path, output_path: Path) -> None:
    """
    Decompress a gzipped file.
    
    Parameters
    ----------
    gz_path : Path
        Path to .gz file
    output_path : Path
        Path for decompressed output
    """
    print(f"📦 Decompressing {gz_path.name}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"   ✓ Decompressed to {output_path.name}")


def mouse_brain_5k_atacseq(force_download: bool = False, use_local: bool = True) -> Tuple[Path, Path]:
    """
    Download 10X Mouse Brain 5k scATAC-seq dataset and mouse annotation.
    
    This dataset contains ~5,000 cells from mouse brain tissue, profiled
    with 10X Chromium scATAC-seq. Automatically downloads both the peak
    matrix (H5 format) and the mouse GENCODE annotation (GTF).
    
    Parameters
    ----------
    force_download : bool, optional
        If True, re-download even if files exist (default: False)
    use_local : bool, optional
        If True, check examples/data/ for files before downloading (default: True)
    
    Returns
    -------
    h5_path : Path
        Path to filtered_peak_bc_matrix.h5 file
    gtf_path : Path
        Path to gencode.vM25.annotation.gtf file
    
    Examples
    --------
    >>> import iaode
    >>> h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()
    >>> adata = iaode.annotation_pipeline(h5_file, gtf_file)
    
    Notes
    -----
    If automatic download fails (403 error), the function will look for files in:
    - Current directory: examples/data/
    - User cache: ~/.iaode/data/
    
    Manual download instructions will be provided if files are not found.
    """
    data_dir = get_data_dir()
    atacseq_dir = data_dir / "atacseq"
    anno_dir = data_dir / "annotations"
    
    # Define file paths
    h5_file = atacseq_dir / "mouse_brain_5k_v1.1.h5"
    gtf_file = anno_dir / "gencode.vM25.annotation.gtf"
    gtf_gz = anno_dir / "gencode.vM25.annotation.gtf.gz"
    
    # Check for local files first (e.g., in examples/data/)
    if use_local:
        local_examples_dir = Path.cwd() / "data"
        if not local_examples_dir.exists():
            local_examples_dir = Path.cwd().parent / "data"  # Try parent dir
        
        local_h5 = local_examples_dir / "mouse_brain_5k_v1.1.h5"
        local_gtf = local_examples_dir / "gencode.vM25.annotation.gtf"
        
        if local_h5.exists() and local_gtf.exists():
            print(f"✓ Using local files from: {local_examples_dir}")
            # Copy to cache for consistency
            if not h5_file.exists():
                print(f"  Copying H5 to cache...")
                shutil.copy(local_h5, h5_file)
            if not gtf_file.exists():
                print(f"  Copying GTF to cache...")
                shutil.copy(local_gtf, gtf_file)
            print()
            print("=" * 70)
            print("✅ Dataset ready!")
            print("=" * 70)
            print(f"H5 file:  {h5_file}")
            print(f"GTF file: {gtf_file}")
            print(f"Cache location: {data_dir}")
            print("=" * 70)
            print()
            return h5_file, gtf_file
    
    # Download H5 file if needed
    if not h5_file.exists() or force_download:
        h5_url = "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_mouse_brain_5k_v1.1/atac_mouse_brain_5k_v1.1_filtered_peak_bc_matrix.h5"
        try:
            _download_file(h5_url, h5_file, "mouse brain 5k scATAC-seq H5 matrix")
        except Exception as e:
            print(f"\n⚠️  Automatic download failed: {e}")
            print("\n" + "=" * 70)
            print("MANUAL DOWNLOAD REQUIRED")
            print("=" * 70)
            print("Please download the file manually:")
            print(f"\n1. Visit: https://www.10xgenomics.com/datasets/fresh-cortex-from-adult-mouse-brain-p-50-1-standard-2-0-0")
            print("   Or try direct link:")
            print(f"   {h5_url}")
            print(f"\n2. Save as: {h5_file}")
            print(f"\n3. Or place in: {Path.cwd() / 'data' / 'mouse_brain_5k_v1.1.h5'}")
            print("=" * 70)
            raise
    else:
        print(f"✓ Using cached H5: {h5_file}")
    
    # Download GTF if needed
    if not gtf_file.exists() or force_download:
        if not gtf_gz.exists() or force_download:
            gtf_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz"
            _download_file(gtf_url, gtf_gz, "GENCODE vM25 mouse annotation (847 MB compressed)")
        
        # Decompress
        _decompress_gz(gtf_gz, gtf_file)
        
        # Clean up compressed file to save space
        if gtf_gz.exists():
            gtf_gz.unlink()
            print(f"   ✓ Removed compressed file to save space")
    else:
        print(f"✓ Using cached GTF: {gtf_file}")
    
    print()
    print("=" * 70)
    print("✅ Dataset ready!")
    print("=" * 70)
    print(f"H5 file:  {h5_file}")
    print(f"GTF file: {gtf_file}")
    print(f"Cache location: {data_dir}")
    print("=" * 70)
    print()
    
    return h5_file, gtf_file


def human_pbmc_5k_atacseq(force_download: bool = False) -> Tuple[Path, Path]:
    """
    Download 10X Human PBMC 5k scATAC-seq dataset and human annotation.
    
    This dataset contains ~5,000 peripheral blood mononuclear cells (PBMCs),
    profiled with 10X Chromium scATAC-seq. Automatically downloads both the
    peak matrix (H5 format) and the human GENCODE annotation (GTF).
    
    Parameters
    ----------
    force_download : bool, optional
        If True, re-download even if files exist (default: False)
    
    Returns
    -------
    h5_path : Path
        Path to filtered_peak_bc_matrix.h5 file
    gtf_path : Path
        Path to gencode.v49.annotation.gtf file
    
    Examples
    --------
    >>> import iaode
    >>> h5_file, gtf_file = iaode.datasets.human_pbmc_5k_atacseq()
    >>> adata = iaode.annotation_pipeline(h5_file, gtf_file)
    """
    data_dir = get_data_dir()
    atacseq_dir = data_dir / "atacseq"
    anno_dir = data_dir / "annotations"
    
    # Define file paths
    h5_file = atacseq_dir / "human_pbmc_5k_nextgem.h5"
    gtf_file = anno_dir / "gencode.v49.annotation.gtf"
    gtf_gz = anno_dir / "gencode.v49.annotation.gtf.gz"
    
    # Download H5 file if needed
    if not h5_file.exists() or force_download:
        h5_url = "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_filtered_peak_bc_matrix.h5"
        _download_file(h5_url, h5_file, "human PBMC 5k scATAC-seq H5 matrix")
    else:
        print(f"✓ Using cached H5: {h5_file}")
    
    # Download GTF if needed
    if not gtf_file.exists() or force_download:
        if not gtf_gz.exists() or force_download:
            gtf_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz"
            _download_file(gtf_url, gtf_gz, "GENCODE v49 human annotation (1.4 GB compressed)")
        
        # Decompress
        _decompress_gz(gtf_gz, gtf_file)
        
        # Clean up compressed file
        if gtf_gz.exists():
            gtf_gz.unlink()
            print(f"   ✓ Removed compressed file to save space")
    else:
        print(f"✓ Using cached GTF: {gtf_file}")
    
    print()
    print("=" * 70)
    print("✅ Dataset ready!")
    print("=" * 70)
    print(f"H5 file:  {h5_file}")
    print(f"GTF file: {gtf_file}")
    print(f"Cache location: {data_dir}")
    print("=" * 70)
    print()
    
    return h5_file, gtf_file


def clear_cache() -> None:
    """
    Clear the iAODE data cache directory.
    
    Removes all downloaded datasets to free disk space.
    Files will be re-downloaded on next use.
    """
    data_dir = get_data_dir()
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"✓ Cleared cache: {data_dir}")
        print("  Files will be re-downloaded when needed.")
    else:
        print("✓ Cache already empty")


def list_cached_files() -> None:
    """
    List all cached dataset files and their sizes.
    """
    data_dir = get_data_dir()
    print("=" * 70)
    print("iAODE Cached Datasets")
    print("=" * 70)
    print(f"Location: {data_dir}")
    print()
    
    total_size: float = 0.0
    file_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            filepath = Path(root) / file
            size_mb = filepath.stat().st_size / 1024 / 1024
            total_size += size_mb
            file_count += 1
            rel_path = filepath.relative_to(data_dir)
            print(f"  {rel_path}")
            print(f"    Size: {size_mb:.1f} MB")
    
    print()
    print(f"Total: {file_count} files, {total_size:.1f} MB")
    print("=" * 70)
    
    if file_count > 0:
        print("\nTo clear cache: iaode.datasets.clear_cache()")
