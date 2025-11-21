"""
scATAC-seq Peak Annotation Example

Complete pipeline for annotating scATAC-seq peaks to genes using
genomic features from GTF files.

Data is automatically downloaded to ~/.iaode/data/ on first run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _example_utils import (
    check_iaode_installed, setup_output_dir,
    print_header, print_section, print_success, print_info
)

if not check_iaode_installed():
    sys.exit(1)

import iaode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = setup_output_dir("atacseq_annotation")

# ==================================================
# Download Data (Automatic)
# ==================================================

print_header("scATAC-seq Peak Annotation Pipeline")

print_section("Downloading data (automatic)")
print_info("Dataset: 10X Mouse Brain 5k scATAC-seq + GENCODE vM25 annotation")
print_info("Files will be cached in: ~/.iaode/data/")
print()

# Automatically download data if not cached
h5_file, gtf_file = iaode.datasets.mouse_brain_5k_atacseq()

print_success("Data ready for analysis")
print_info(f"  H5 file:  {h5_file.name}")
print_info(f"  GTF file: {gtf_file.name}")
print()

# ==================================================
# Run Annotation Pipeline
# ==================================================

print_section("Running annotation pipeline")
print_info("Pipeline configuration:")
print("  • Promoter region: -2000 to +500 bp from TSS")
print("  • TF-IDF normalization: Enabled")
print("  • Highly variable peaks (HVP): Top 20,000")
print()

adata = iaode.annotation_pipeline(
    h5_file=str(h5_file),
    gtf_file=str(gtf_file),
    promoter_upstream=2000,
    promoter_downstream=500,
    apply_tfidf=True,
    select_hvp=True,
    n_top_peaks=20000
)

print_success(f"Pipeline complete: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
print()

# ==================================================
# Quick Statistics
# ==================================================

print_section("Computing annotation statistics")

# Peak counts per cell
try:
    peak_counts_mat = adata.X.sum(axis=1)  # type: ignore[call-arg]
    if hasattr(peak_counts_mat, 'A1'):
        peak_counts = peak_counts_mat.A1
    else:
        peak_counts = np.asarray(peak_counts_mat).ravel()
except Exception:
    peak_counts = np.asarray(adata.X).sum(axis=1)

# Distance to TSS statistics
if 'distance_to_tss' in adata.var.columns:
    distances_array = np.asarray(adata.var['distance_to_tss'])
    distances_finite = distances_array[np.isfinite(distances_array)]
else:
    distances_finite = np.array([])

print_info("Peak statistics:")
print(f"  Total peaks: {adata.n_vars:,}")
print(f"  Mean peaks per cell: {peak_counts.mean():.1f} ± {peak_counts.std():.1f}")
print(f"  Median peaks per cell: {np.median(peak_counts):.1f}")
if len(distances_finite) > 0:
    print(f"  Mean distance to TSS: {np.abs(distances_finite).mean():.1f} bp")
    print(f"  Median distance to TSS: {np.median(np.abs(distances_finite)):.1f} bp")
print()

# ==================================================
# Visualization
# ==================================================

print_section("Generating visualizations")

# Set global style matching previous examples
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Ubuntu', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Professional color palette - Wong's colorblind-friendly palette with high contrast
COLORS = {
    # Genomic feature colors (highly distinct and colorblind-safe)
    'gene_body': '#56B4E9',   # Sky Blue - main gene structure (54.4%)
    'intergenic': '#E69F00',  # Orange - between genes (20.0%)
    'distal': '#CC79A7',      # Reddish Purple - enhancers/distal elements (16.5%)
    'promoter': '#D55E00',    # Vermillion - critical regulatory regions (9.0%)
    
    # Alternative names
    'genebody': '#56B4E9',
    'gene body': '#56B4E9',
    
    # For histograms and other plots
    'blue': '#0077BB',
    'orange': '#E69F00',
    'teal': '#009988',
    
    # Statistical lines
    'mean': '#D55E00',        # Vermillion for mean
    'median': '#009988',      # Teal for median
    'hvp': '#009988',         # Teal for highlighted features
    'grey': '#CCCCCC',        # Light grey for non-selected
}

# Create figure with 2×2 grid
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.08, right=0.96,
                       top=0.94, bottom=0.08,
                       hspace=0.35, wspace=0.30)

def style_axis(ax, grid=True):
    """Apply consistent styling to axes"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)
    if grid:
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)

# Panel A: Peak Annotation Distribution (Pie Chart)
ax1 = fig.add_subplot(gs[0, 0])

if 'annotation_type' in adata.var.columns:
    annotation_counts = adata.var['annotation_type'].value_counts()
    
    # Map annotation types to colors (case-insensitive)
    color_map = {
        'gene_body': COLORS['gene_body'],
        'genebody': COLORS['gene_body'],
        'gene body': COLORS['gene_body'],
        'intergenic': COLORS['intergenic'],
        'distal': COLORS['distal'],
        'promoter': COLORS['promoter'],
        'promoter-tss': COLORS['promoter'],
        'promoter_tss': COLORS['promoter'],
    }
    
    colors = [color_map.get(str(ann).lower().replace(' ', '_').replace('-', '_'), 
                            COLORS['grey']) for ann in annotation_counts.index]
    
    # Create pie chart with enhanced styling
    wedges, texts, autotexts = ax1.pie(
        annotation_counts.values,
        labels=[str(label).replace('_', ' ').title() for label in annotation_counts.index],
        autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
        textprops={'fontsize': 10}
    )
    
    # Style text for better readability
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax1.set_title('A. Peak Annotation Distribution', 
                 fontsize=12, fontweight='bold', loc='left', pad=15)
else:
    ax1.text(0.5, 0.5, 'Annotation data\nnot available',
            ha='center', va='center', fontsize=12, 
            color='#666666', style='italic')
    ax1.set_title('A. Peak Annotation Distribution',
                 fontsize=12, fontweight='bold', loc='left', pad=15)
    ax1.axis('off')

# Panel B: Distance to TSS Distribution
ax2 = fig.add_subplot(gs[0, 1])

if 'distance_to_tss' in adata.var.columns and len(distances_finite) > 0:
    # Filter extreme values for better visualization
    distances_plot = distances_finite[np.abs(distances_finite) < 50000]
    
    if len(distances_plot) > 0:
        counts, bins, patches = ax2.hist(
            distances_plot / 1000,  # Convert to kb
            bins=50,
            color=COLORS['blue'],
            alpha=0.75,
            edgecolor='black',
            linewidth=0.8
        )
        
        # Add mean and median lines
        mean_dist = np.mean(distances_plot) / 1000
        median_dist = np.median(distances_plot) / 1000
        
        ax2.axvline(mean_dist, color=COLORS['mean'],
                   linestyle='--', linewidth=2.5,
                   label=f'Mean: {mean_dist:.1f} kb', zorder=3)
        ax2.axvline(median_dist, color=COLORS['median'],
                   linestyle='--', linewidth=2.5,
                   label=f'Median: {median_dist:.1f} kb', zorder=3)
        
        ax2.set_xlabel('Distance to TSS (kb)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Peak Count', fontsize=11, fontweight='bold')
        ax2.set_title('B. Peak Distance to TSS',
                     fontsize=12, fontweight='bold', loc='left', pad=10)
        ax2.legend(fontsize=9, frameon=True, edgecolor='black', 
                  framealpha=0.95, loc='upper right')
        style_axis(ax2)
    else:
        ax2.text(0.5, 0.5, 'No valid TSS distances',
                ha='center', va='center', fontsize=12,
                color='#666666', style='italic')
        ax2.set_title('B. Peak Distance to TSS',
                     fontsize=12, fontweight='bold', loc='left', pad=10)
        ax2.axis('off')
else:
    ax2.text(0.5, 0.5, 'Distance to TSS\nnot available',
            ha='center', va='center', fontsize=12,
            color='#666666', style='italic')
    ax2.set_title('B. Peak Distance to TSS',
                 fontsize=12, fontweight='bold', loc='left', pad=10)
    ax2.axis('off')

# Panel C: Peak Counts per Cell
ax3 = fig.add_subplot(gs[1, 0])

counts, bins, patches = ax3.hist(
    peak_counts,
    bins=50,
    color=COLORS['orange'],
    alpha=0.75,
    edgecolor='black',
    linewidth=0.8
)

# Add statistics lines
mean_counts = peak_counts.mean()
median_counts = np.median(peak_counts)

ax3.axvline(mean_counts, color=COLORS['mean'],
           linestyle='--', linewidth=2.5,
           label=f'Mean: {mean_counts:.0f}', zorder=3)
ax3.axvline(median_counts, color=COLORS['median'],
           linestyle='--', linewidth=2.5,
           label=f'Median: {median_counts:.0f}', zorder=3)

ax3.set_xlabel('Peak Counts per Cell', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
ax3.set_title('C. Peak Coverage per Cell',
             fontsize=12, fontweight='bold', loc='left', pad=10)
ax3.legend(fontsize=9, frameon=True, edgecolor='black',
          framealpha=0.95, loc='upper right')
style_axis(ax3)

# Panel D: Highly Variable Peaks Selection
ax4 = fig.add_subplot(gs[1, 1])

if 'highly_variable' in adata.var.columns:
    hvp_count = adata.var['highly_variable'].sum()
    non_hvp_count = len(adata.var) - hvp_count
    
    categories = ['Highly Variable\nPeaks (HVP)', 'Other Peaks']
    counts_bar = [hvp_count, non_hvp_count]
    colors_bar = [COLORS['hvp'], COLORS['grey']]
    
    bars = ax4.bar(categories, counts_bar, 
                   color=colors_bar,
                   edgecolor='black',
                   linewidth=1.2,
                   alpha=0.85)
    
    ax4.set_ylabel('Number of Peaks', fontsize=11, fontweight='bold')
    ax4.set_title('D. Highly Variable Peaks Selection',
                 fontsize=12, fontweight='bold', loc='left', pad=10)
    style_axis(ax4)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts_bar)):
        height = bar.get_height()
        pct = count / sum(counts_bar) * 100
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color='black')
else:
    ax4.text(0.5, 0.5, 'HVP selection\nnot performed',
            ha='center', va='center', fontsize=12,
            color='#666666', style='italic')
    ax4.set_title('D. Highly Variable Peaks Selection',
                 fontsize=12, fontweight='bold', loc='left', pad=10)
    ax4.axis('off')

# Save figure
plt.savefig(OUTPUT_DIR / 'annotation_qc.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'annotation_qc.pdf', dpi=300, bbox_inches='tight')
plt.close()

print_success(f"Saved: {OUTPUT_DIR}/annotation_qc.png")
print_success(f"Saved: {OUTPUT_DIR}/annotation_qc.pdf")
print()

# ==================================================
# Save Results
# ==================================================

print_section("Saving annotated data")

output_file = OUTPUT_DIR / "annotated_peaks.h5ad"
adata.write_h5ad(output_file)
print_success(f"Saved: {output_file}")
print()

# ==================================================
# Detailed Summary
# ==================================================

print_header("Annotation Summary")

print_info("Dataset Overview:")
print(f"  Total peaks: {adata.n_vars:,}")
print(f"  Total cells: {adata.n_obs:,}")
print()

if 'annotation_type' in adata.var.columns:
    print_info("Genomic Annotation Distribution:")
    annotation_type_counts = adata.var['annotation_type'].value_counts()
    
    # Use colored symbols and emojis for better visualization
    symbols = {
        'gene_body': '🔵',      # Blue circle for gene body
        'genebody': '🔵',
        'intergenic': '🟠',     # Orange circle for intergenic
        'distal': '🟣',         # Purple circle for distal
        'promoter': '🔴',       # Red circle for promoter
    }
    
    for annotation_type, count in annotation_type_counts.items():
        pct = count / len(adata.var) * 100
        # Create a visual bar
        bar_length = int(pct / 2)  # Scale to max 50 chars
        bar = '█' * bar_length
        symbol = symbols.get(str(annotation_type).lower().replace(' ', '_'), '○')
        label = str(annotation_type).replace('_', ' ').title()
        print(f"  {symbol} {label:15s}: {count:6,} ({pct:5.1f}%) {bar}")
    print()

if 'highly_variable' in adata.var.columns:
    hvp_count = adata.var['highly_variable'].sum()
    hvp_pct = hvp_count / len(adata.var) * 100
    print_info("Feature Selection:")
    print(f"  🟢 Highly variable peaks: {hvp_count:,} ({hvp_pct:.1f}%)")
    print(f"  ⚪ Other peaks:           {len(adata.var) - hvp_count:,} ({100-hvp_pct:.1f}%)")
    print()

print_info("Cell Quality Metrics:")
print(f"  Mean peaks per cell:   {peak_counts.mean():.1f} ± {peak_counts.std():.1f}")
print(f"  Median peaks per cell: {np.median(peak_counts):.1f}")
print(f"  Range:                 [{peak_counts.min():.0f}, {peak_counts.max():.0f}]")
print()

if len(distances_finite) > 0:
    print_info("Genomic Distance Statistics:")
    print(f"  Peaks with TSS annotation: {len(distances_finite):,} ({len(distances_finite)/len(adata.var)*100:.1f}%)")
    print(f"  Mean distance to TSS:      {np.abs(distances_finite).mean():.1f} bp")
    print(f"  Median distance to TSS:    {np.median(np.abs(distances_finite)):.1f} bp")
    print()

print_header("Pipeline Complete")
print_info("Next steps:")
print("  1. Use annotated data with iaode.agent() for dimensionality reduction")
print("  2. Perform trajectory inference with use_ode=True")
print("  3. Link peaks to genes for gene activity scoring")
print()
print_info("Example usage:")
print("  >>> import scanpy as sc")
print("  >>> import iaode")
print("  >>> adata = sc.read_h5ad('annotated_peaks.h5ad')")
print("  >>> model = iaode.agent(adata, layer='counts', use_ode=True)")
print("  >>> model.fit(epochs=100)")
print()