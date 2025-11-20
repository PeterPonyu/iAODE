#!/bin/bash

# Download helper for iAODE scATAC-seq examples
# This script downloads verified reference GTF and 10X scATAC-seq sample data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "iAODE scATAC-seq Example Data Downloader"
echo "============================================================"
echo

# Parse command line arguments
SPECIES="${1:-human}"
DATASET="${2:-5k_pbmc}"

case "$SPECIES" in
  human)
    echo "📥 Downloading Human GENCODE v49 (GRCh38/hg38)..."
    GTF_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz"
    GTF_FILE="gencode.v49.annotation.gtf.gz"
    ;;
  mouse)
    echo "📥 Downloading Mouse GENCODE vM25 (GRCm38/mm10)..."
    GTF_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz"
    GTF_FILE="gencode.vM25.annotation.gtf.gz"
    ;;
  *)
    echo "❌ Error: Unknown species '$SPECIES'"
    echo "Usage: $0 [human|mouse] [5k_pbmc|10k_pbmc|8k_cortex]"
    exit 1
    ;;
esac

# Download GTF if not exists
if [ -f "$GTF_FILE" ]; then
  echo "   ✓ GTF already exists: $GTF_FILE"
else
  wget -O "$GTF_FILE" "$GTF_URL"
  echo "   ✓ Downloaded: $GTF_FILE"
fi

echo

# Download 10X scATAC-seq sample
case "$DATASET" in
  5k_pbmc)
    echo "📥 Downloading 10X 5k Human PBMCs (ATAC v1.1)..."
    BASE_URL="https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem"
    H5_FILE="atac_pbmc_5k_nextgem_filtered_peak_bc_matrix.h5"
    ;;
  10k_pbmc)
    echo "📥 Downloading 10X 10k Human PBMCs (ATAC v2)..."
    BASE_URL="https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_pbmc_10k_v2"
    H5_FILE="atac_pbmc_10k_v2_filtered_peak_bc_matrix.h5"
    ;;
  8k_cortex)
    echo "📥 Downloading 10X 8k Mouse Cortex (ATAC v2)..."
    BASE_URL="https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_mouse_cortex_8k_v2"
    H5_FILE="atac_mouse_cortex_8k_v2_filtered_peak_bc_matrix.h5"
    ;;
  *)
    echo "❌ Error: Unknown dataset '$DATASET'"
    echo "Available datasets: 5k_pbmc, 10k_pbmc, 8k_cortex"
    exit 1
    ;;
esac

# Try to download H5 file (some URLs may require browsing)
if [ -f "$H5_FILE" ]; then
  echo "   ✓ H5 file already exists: $H5_FILE"
else
  echo "   Attempting to download from: ${BASE_URL}/${H5_FILE}"
  if wget -q --spider "${BASE_URL}/${H5_FILE}"; then
    wget -O "$H5_FILE" "${BASE_URL}/${H5_FILE}"
    echo "   ✓ Downloaded: $H5_FILE"
  else
    echo "   ⚠️  Direct download not available. Please visit:"
    echo "      $BASE_URL"
    echo "   and download the 'filtered_peak_bc_matrix.h5' file manually."
  fi
fi

echo
echo "============================================================"
echo "✅ Download complete!"
echo "============================================================"
echo "Files in: $SCRIPT_DIR"
echo
echo "Next steps:"
echo "  1. Update paths in ../atacseq_annotation.py to match downloaded files"
echo "  2. Run: python ../atacseq_annotation.py"
echo
