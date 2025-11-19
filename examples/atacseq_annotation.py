"""
scATAC-seq Peak Annotation Example

This example demonstrates the complete pipeline for annotating
scATAC-seq peaks to genes using genomic features.
"""

import iaode

# File paths (replace with your data)
H5_FILE = "data/filtered_peak_bc_matrix.h5"
GTF_FILE = "data/gencode.v44.annotation.gtf.gz"
OUTPUT_FILE = "results/annotated_peaks.h5ad"

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
print(f"\n📈 Peak Accessibility:")
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
import seaborn as sns

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
print(f"\n💾 QC plot saved to 'peak_annotation_qc.png'")

print("\n✅ Done! Annotated data saved to:", OUTPUT_FILE)
