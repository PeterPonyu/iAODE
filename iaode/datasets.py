
"""
Dataset utilities for iAODE examples.

Provides automatic download and caching of example datasets,
including scATAC-seq data and reference annotations.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List
import requests
from tqdm import tqdm


def get_data_dir() -> Path:
    """Get the iAODE data cache directory (fallback only)."""
    data_dir = Path.home() / ".iaode" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "atacseq").mkdir(exist_ok=True)
    (data_dir / "annotations").mkdir(exist_ok=True)
    return data_dir


def _download_file(url: str, output_path: Path, desc: str = "file") -> bool:
    """
    Download a file with progress indication.
    
    Returns
    -------
    bool
        True if download succeeded, False otherwise
    """
    print(f"📥 Attempting download: {desc}")
    print(f"   URL: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(output_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    size = f.write(chunk)
                    pbar.update(size)
        
        print("   ✓ Downloaded successfully")
        return True
        
    except Exception as e:
        print(f"   ✗ Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def _try_multiple_urls(urls: List[str], output_path: Path, desc: str) -> bool:
    """Try downloading from multiple URLs until one succeeds."""
    for i, url in enumerate(urls, 1):
        print(f"\n[Attempt {i}/{len(urls)}]")
        if _download_file(url, output_path, desc):
            return True
    return False


def _decompress_gz(gz_path: Path, output_path: Path) -> None:
    """Decompress a gzipped file."""
    import gzip
    print(f"📦 Decompressing {gz_path.name}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"   ✓ Decompressed to {output_path.name}")


def mouse_brain_5k_atacseq(force_download: bool = False) -> Tuple[Path, Path]:
    """
    Get 10X Mouse Brain 5k scATAC-seq dataset and mouse annotation.
    
    Priority order:
    1. Local data/ folder (current or parent directory)
    2. Cache (~/.iaode/data/)
    3. Download to cache
    
    Parameters
    ----------
    force_download : bool
        Re-download to cache even if files exist
    
    Returns
    -------
    h5_path, gtf_path : Tuple[Path, Path]
        Paths to H5 and GTF files
    """
    # Step 1: Check local data folders first
    local_search_dirs = [
        Path.cwd() / "data",
        Path.cwd().parent / "data",
        Path.cwd() / "examples" / "data"
    ]
    
    for local_dir in local_search_dirs:
        local_h5 = local_dir / "mouse_brain_5k_v1.1.h5"
        local_gtf = local_dir / "gencode.vM25.annotation.gtf"
        
        if local_h5.exists() and local_gtf.exists():
            print("=" * 70)
            print("✅ Using local files (no download needed)")
            print("=" * 70)
            print(f"Location: {local_dir}")
            print(f"H5:  {local_h5.name}")
            print(f"GTF: {local_gtf.name}")
            print("=" * 70 + "\n")
            return local_h5, local_gtf
    
    # Step 2: Check cache if not forced to re-download
    data_dir = get_data_dir()
    h5_file = data_dir / "atacseq" / "mouse_brain_5k_v1.1.h5"
    gtf_file = data_dir / "annotations" / "gencode.vM25.annotation.gtf"
    gtf_gz = data_dir / "annotations" / "gencode.vM25.annotation.gtf.gz"
    
    if not force_download and h5_file.exists() and gtf_file.exists():
        print(f"✓ Using cached files: {data_dir}")
        return h5_file, gtf_file
    
    # Step 3: Download to cache
    print("Local files not found. Downloading to cache...")
    
    # Download H5 if needed
    if not h5_file.exists() or force_download:
        h5_urls = [
            "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_mouse_brain_5k_v1.1/atac_mouse_brain_5k_v1.1_filtered_peak_bc_matrix.h5",
            "https://cf.10xgenomics.com/samples/cell-atac/1.2.0/atac_mouse_brain_5k_v1.1/atac_mouse_brain_5k_v1.1_filtered_peak_bc_matrix.h5",
        ]
        
        if not _try_multiple_urls(h5_urls, h5_file, "Mouse Brain 5k H5"):
            print("\n" + "=" * 70)
            print("⚠️  AUTOMATIC DOWNLOAD FAILED")
            print("=" * 70)
            print("Please download manually:")
            print("\n1. Visit: https://www.10xgenomics.com/datasets")
            print("   Search: 'Fresh Cortex from Adult Mouse Brain'")
            print(f"\n2. Download 'Filtered peak-barcode matrix HDF5'")
            print(f"\n3. Save to one of these locations:")
            print(f"   • {Path.cwd() / 'data' / 'mouse_brain_5k_v1.1.h5'}")
            print(f"   • {h5_file}")
            print("=" * 70)
            raise RuntimeError("Manual download required")
    
    # Download GTF if needed
    if not gtf_file.exists() or force_download:
        if not gtf_gz.exists() or force_download:
            gtf_urls = [
                "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz",
                "ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz"
            ]
            
            if not _try_multiple_urls(gtf_urls, gtf_gz, "GENCODE vM25 GTF"):
                print("\n⚠️  GTF download failed. Please download manually:")
                print(f"   URL: {gtf_urls[0]}")
                print(f"   Save to: {Path.cwd() / 'data' / 'gencode.vM25.annotation.gtf.gz'}")
                raise RuntimeError("Manual GTF download required")
        
        _decompress_gz(gtf_gz, gtf_file)
        if gtf_gz.exists():
            gtf_gz.unlink()
    
    print("\n" + "=" * 70)
    print("✅ Dataset ready (cached)")
    print("=" * 70)
    print(f"Cache: {data_dir}")
    print(f"H5:  {h5_file.name}")
    print(f"GTF: {gtf_file.name}")
    print("=" * 70 + "\n")
    
    return h5_file, gtf_file


def human_pbmc_5k_atacseq(force_download: bool = False) -> Tuple[Path, Path]:
    """
    Get 10X Human PBMC 5k scATAC-seq dataset.
    
    Priority: local data/ folder → cache → download
    """
    # Check local first
    local_search_dirs = [
        Path.cwd() / "data",
        Path.cwd().parent / "data",
        Path.cwd() / "examples" / "data"
    ]
    
    for local_dir in local_search_dirs:
        local_h5 = local_dir / "human_pbmc_5k_nextgem.h5"
        local_gtf = local_dir / "gencode.v49.annotation.gtf"
        
        if local_h5.exists() and local_gtf.exists():
            print(f"✓ Using local files: {local_dir}")
            return local_h5, local_gtf
    
    # Check cache
    data_dir = get_data_dir()
    h5_file = data_dir / "atacseq" / "human_pbmc_5k_nextgem.h5"
    gtf_file = data_dir / "annotations" / "gencode.v49.annotation.gtf"
    gtf_gz = data_dir / "annotations" / "gencode.v49.annotation.gtf.gz"
    
    if not force_download and h5_file.exists() and gtf_file.exists():
        print(f"✓ Using cached files: {data_dir}")
        return h5_file, gtf_file
    
    # Download to cache
    print("Local files not found. Downloading to cache...")
    
    if not h5_file.exists() or force_download:
        h5_urls = [
            "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_filtered_peak_bc_matrix.h5",
            "https://cf.10xgenomics.com/samples/cell-atac/1.1.0/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_filtered_peak_bc_matrix.h5",
        ]
        
        if not _try_multiple_urls(h5_urls, h5_file, "PBMC 5k H5"):
            print("\n⚠️  Download failed. Save manually to:")
            print(f"   {Path.cwd() / 'data' / 'human_pbmc_5k_nextgem.h5'}")
            raise RuntimeError("Manual download required")
    
    if not gtf_file.exists() or force_download:
        if not gtf_gz.exists() or force_download:
            gtf_urls = [
                "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz",
            ]
            _try_multiple_urls(gtf_urls, gtf_gz, "GENCODE v49 GTF")
        
        _decompress_gz(gtf_gz, gtf_file)
        if gtf_gz.exists():
            gtf_gz.unlink()
    
    print(f"\n✅ Dataset ready: {h5_file.name}, {gtf_file.name}\n")
    return h5_file, gtf_file


def clear_cache() -> None:
    """Clear the iAODE data cache (~/.iaode/data/)."""
    data_dir = get_data_dir()
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"✓ Cleared cache: {data_dir}")
    else:
        print("✓ Cache already empty")


def list_cached_files() -> None:
    """List all cached dataset files."""
    data_dir = get_data_dir()
    print("=" * 70)
    print("iAODE Cached Datasets")
    print("=" * 70)
    print(f"Cache location: {data_dir}")
    print("\nNote: Local data/ folders are checked first and not listed here.\n")
    
    if not data_dir.exists():
        print("Cache is empty.\n")
        print("=" * 70)
        return
    
    total_size = 0.0
    file_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            filepath = Path(root) / file
            size_mb = filepath.stat().st_size / 1024 / 1024
            total_size += size_mb
            file_count += 1
            print(f"  {filepath.relative_to(data_dir)}")
            print(f"    {size_mb:.1f} MB")
    
    if file_count == 0:
        print("Cache is empty.\n")
    else:
        print(f"\nTotal: {file_count} files, {total_size:.1f} MB")
    
    print("=" * 70)
