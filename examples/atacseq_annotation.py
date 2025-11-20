"""
scATAC-seq Peak Annotation Example

This example demonstrates the complete pipeline for annotating
scATAC-seq peaks to genes using genomic features.
"""

import iaode
from pathlib import Path
import sys

# Recommend placing example data under the examples/data/ directory.
# This keeps examples portable and avoids hard-coded absolute paths.
EXAMPLE_DIR = Path(__file__).parent
DATA_DIR = EXAMPLE_DIR / "data"

# File names (update these to match the files you put in examples/data/)
H5_FILE = DATA_DIR / "filtered_peak_bc_matrix.h5"
GTF_FILE = DATA_DIR / "gencode.v44.annotation.gtf.gz"
OUTPUT_FILE = EXAMPLE_DIR / "results" / "annotated_peaks.h5ad"

# Ensure data files exist and give clear instructions if not
if not H5_FILE.exists() or not GTF_FILE.exists():
    print("\nERROR: Required example data not found.")
    print(f"  Expected H5 file at: {H5_FILE}")
    print(f"  Expected GTF file at: {GTF_FILE}\n")
    print("Please place your 10X scATAC 'filtered_peak_bc_matrix.h5' and the GTF file")
    print("under the 'examples/data/' folder, or update the paths at the top of this script.")

    print("\nVerified reference downloads (recommended):")
    print("  GENCODE GTFs:")
    print("    - Human v19 (GRCh37/hg19): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz")
    print("    - Mouse vM25 (GRCm38/mm10): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz")
    print("    - Human v49 (GRCh38/hg38): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz")
    print("    - Mouse vM38 (GRCm39/mm39): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.annotation.gtf.gz")

    print("  10X Genomics scATAC-seq example datasets (base URLs):")
    print("    - 5k Human PBMCs (ATAC v1.1): https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/")
    print("    - 10k Human PBMCs (ATAC v2): https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_pbmc_10k_v2/")
    print("    - 8k Mouse Cortex (ATAC v2): https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_mouse_cortex_8k_v2/")

    print("Quick download examples:")
    print("  # Download a GENCODE GTF (example: human v49)")
    print("  wget -P examples/data/ https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz")
    print("  # Example: download 10X filtered peak matrix (open the base URL and pick the appropriate file, e.g. 'filtered_peak_bc_matrix.h5')")
    print("  wget -P examples/data/ https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/filtered_peak_bc_matrix.h5")

    print("Note: File names at 10X sample directories may vary. If a direct 'filtered_peak_bc_matrix.h5' is not present, visit the base URL in a browser and download the appropriate archive or H5 file.")
    print("\nNote: Large GTF files (e.g. ~800 MB) should NOT be pushed directly to the repository. Prefer hosting the files externally (Zenodo, S3, figshare) and providing a small download helper script in 'examples/data/' to fetch them.")
    sys.exit(1)

print("="*70)
print("scATAC-seq Peak Annotation Pipeline")
print("="*70)

# Run complete annotation pipeline
adata = iaode.annotation_pipeline(
    h5_file=H5_FILE,
    gtf_file=GTF_FILE,
    output_h5ad=OUTPUT_FILE,
    
    # === Peak-to-Gene Annotation Settings ===
    promoter_upstream=2000,        # TSS -2kb
    promoter_downstream=500,       # TSS +500bp
    gene_body=True,                # Include gene body annotations
    distal_threshold=50000,        # 50kb max distance for distal
    gene_type='protein_coding',    # Focus on protein-coding genes
    annotation_priority='promoter', # Prefer promoter over gene body
    
    # === TF-IDF Normalization ===
    apply_tfidf=True,
    tfidf_scale_factor=1e4,        # Standard scale factor
    tfidf_log_tf=False,            # Standard TF (not log-transformed)
    tfidf_log_idf=True,            # Standard IDF (log-transformed)
    
    # === Highly Variable Peaks ===
    select_hvp=True,
    n_top_peaks=20000,             # Select top 20k peaks
    hvp_min_accessibility=0.01,    # Peak in ≥1% of cells
    hvp_max_accessibility=0.95,    # Filter ubiquitous peaks
    hvp_method='signac'            # Use Signac method
)

print("\n" + "="*70)
print("Pipeline Complete!")
print("="*70)

# Inspect results
print(f"\n📊 Dataset Summary:")
print(f"   Cells: {adata.n_obs:,}")
print(f"   Peaks: {adata.n_vars:,}")
print(f"   Highly variable peaks: {adata.var['highly_variable'].sum():,}")

# Annotation statistics
annotation_counts = adata.var['annotation_type'].value_counts()
print(f"\n📍 Peak Annotations:")
for anno_type, count in annotation_counts.items():
    pct = count / adata.n_vars * 100
    print(f"   {anno_type.capitalize():12s}: {count:6,} ({pct:5.1f}%)")

# Top annotated genes
top_genes = adata.var['gene_annotation'].value_counts().head(10)
print(f"\n🧬 Top 10 Annotated Genes:")
for i, (gene, count) in enumerate(top_genes.items(), 1):
    if gene not in ['intergenic', 'parse_failed']:
        print(f"   {i:2d}. {gene:15s}: {count:4d} peaks")

# Accessibility distribution
import numpy as np
print("\nPeak Accessibility:")
print(f"   Mean: {adata.var['accessibility'].mean():.4f}")
print(f"   Median: {adata.var['accessibility'].median():.4f}")
print(f"   Min: {adata.var['accessibility'].min():.4f}")
print(f"   Max: {adata.var['accessibility'].max():.4f}")

# HVP statistics
if 'highly_variable' in adata.var.columns:
    hvp_accessibility = adata.var.loc[
        adata.var['highly_variable'], 'accessibility'
    ].mean()
    print(f"   Mean accessibility (HVPs): {hvp_accessibility:.4f}")

# Quality control plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Annotation type distribution
anno_counts = adata.var['annotation_type'].value_counts()
axes[0, 0].bar(range(len(anno_counts)), anno_counts.values)
axes[0, 0].set_xticks(range(len(anno_counts)))
axes[0, 0].set_xticklabels(anno_counts.index, rotation=45, ha='right')
axes[0, 0].set_ylabel('Number of Peaks')
axes[0, 0].set_title('Peak Annotation Distribution')

# 2. Accessibility histogram
axes[0, 1].hist(adata.var['accessibility'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0.01, color='red', linestyle='--', label='Min threshold')
axes[0, 1].axvline(0.95, color='blue', linestyle='--', label='Max threshold')
axes[0, 1].set_xlabel('Accessibility')
axes[0, 1].set_ylabel('Number of Peaks')
axes[0, 1].set_title('Peak Accessibility Distribution')
axes[0, 1].legend()

# 3. Distance to TSS (log scale)
valid_dist = adata.var['distance_to_tss'].dropna()
axes[1, 0].hist(np.log10(valid_dist + 1), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('log10(Distance to TSS + 1)')
axes[1, 0].set_ylabel('Number of Peaks')
axes[1, 0].set_title('Distance to Nearest TSS')

# 4. Peak width distribution
axes[1, 1].hist(adata.var['peak_width'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Peak Width (bp)')
axes[1, 1].set_ylabel('Number of Peaks')
axes[1, 1].set_title('Peak Width Distribution')

plt.tight_layout()
plt.savefig('peak_annotation_qc.png', dpi=300, bbox_inches='tight')
print("\nQC plot saved to 'peak_annotation_qc.png'")

print("\n✅ Done! Annotated data saved to:", OUTPUT_FILE)
