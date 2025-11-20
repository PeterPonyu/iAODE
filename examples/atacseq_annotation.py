"""
scATAC-seq Peak Annotation Example

Complete pipeline for annotating scATAC-seq peaks to genes using
genomic features from GTF files.

Required data: 10X filtered_peak_bc_matrix.h5 and GENCODE GTF file
Download using: ./examples/data/download_data.sh
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _example_utils import (
    check_iaode_installed, setup_output_dir, check_data_files,
    print_header, print_section, print_success, print_info, print_error
)

if not check_iaode_installed():
    sys.exit(1)

import iaode
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = setup_output_dir("atacseq_annotation")

# ==================================================
# Check Data Files
# ==================================================

print_header("scATAC-seq Peak Annotation Pipeline")

DATA_DIR = Path(__file__).parent / "data"
H5_FILE = DATA_DIR / "mouse_brain_5k_v1.1.h5"
GTF_FILE = DATA_DIR / "gencode.vM25.annotation.gtf"

print_section("Checking data files")

required_files = {
    "H5 peak matrix": H5_FILE,
    "GTF annotation": GTF_FILE
}

if not check_data_files(required_files):
    print_error("Required data files not found!")
    print_info("Download data using:")
    print("  cd examples/data")
    print("  ./download_data.sh mouse 5k_pbmc")
    print()
    print_info("Or manually download:")
    print("  GTF: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz")
    print("  H5:  https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_mouse_brain_5k_v1.1/atac_mouse_brain_5k_v1.1_filtered_peak_bc_matrix.h5")
    sys.exit(1)

print_success("All required data files found")

# ==================================================
# Run Annotation Pipeline
# ==================================================

print_section("Running annotation pipeline")
print_info("Configuration:")
print("  promoter_upstream=2000   → Promoter region definition")
print("  apply_tfidf=True         → TF-IDF normalization for peaks")
print("  select_hvp=True          → Select highly variable peaks")
print("  n_top_peaks=20000        → Number of HVPs to retain")
print()

adata = iaode.annotation_pipeline(
    h5_file=str(H5_FILE),
    gtf_file=str(GTF_FILE),
    promoter_upstream=2000,
    promoter_downstream=500,
    apply_tfidf=True,
    select_hvp=True,
    n_top_peaks=20000
)

print_success(f"Annotated: {adata.n_obs} cells × {adata.n_vars} peaks")

# ==================================================
# Visualize Annotations
# ==================================================

print_section("Generating annotation visualizations")

plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 10})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Peak annotation distribution
if 'peak_type' in adata.var.columns:
    peak_types = adata.var['peak_type'].value_counts()
    ax = axes[0, 0]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D3D3D3']
    ax.pie(peak_types.values, labels=peak_types.index, autopct='%1.1f%%',
           colors=colors[:len(peak_types)], startangle=90)
    ax.set_title('Peak Annotation Distribution', fontweight='bold', pad=20)

# Distance to TSS
if 'distance_to_tss' in adata.var.columns:
    ax = axes[0, 1]
    distances = adata.var['distance_to_tss'].dropna()
    ax.hist(distances[distances.abs() < 50000], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Distance to TSS (bp)')
    ax.set_ylabel('Count')
    ax.set_title('Peak Distance to TSS', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

# Peak counts per cell
ax = axes[1, 0]
peak_counts = adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else adata.X.sum(axis=1)
ax.hist(peak_counts, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
ax.set_xlabel('Peak Counts')
ax.set_ylabel('Number of Cells')
ax.set_title('Peak Counts per Cell', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Highly variable peaks
if 'highly_variable' in adata.var.columns:
    ax = axes[1, 1]
    hvp_count = adata.var['highly_variable'].sum()
    categories = ['HVP', 'Non-HVP']
    counts = [hvp_count, len(adata.var) - hvp_count]
    ax.bar(categories, counts, color=['#F18F01', '#D3D3D3'], edgecolor='black')
    ax.set_ylabel('Number of Peaks')
    ax.set_title('Highly Variable Peaks Selection', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts)*0.02, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'annotation_qc.png', dpi=300, bbox_inches='tight')
plt.close()
print_success(f"Saved: {OUTPUT_DIR}/annotation_qc.png")

# ==================================================
# Save Results
# ==================================================

print_section("Saving annotated data")

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

output_file = results_dir / "annotated_peaks.h5ad"
adata.write_h5ad(output_file)
print_success(f"Saved: {output_file}")

# Summary
print_header("Annotation Summary")
if 'peak_type' in adata.var.columns:
    peak_type_counts = adata.var['peak_type'].value_counts()
    for peak_type, count in peak_type_counts.items():
        pct = count / len(adata.var) * 100
        print(f"  {peak_type:20s}: {count:6d} ({pct:5.1f}%)")

print()
print_info("Annotated data ready for downstream analysis with iAODE")
print_info("Use this data with iaode.agent() for scATAC-seq modeling")
