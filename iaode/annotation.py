
"""
Modern scATAC-seq Peak Annotation Pipeline
Best practices from Signac, SnapATAC2, and scvi-tools (2024)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Literal, Optional, Dict
import re
from collections import defaultdict
import warnings

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================

def load_10x_h5_data(h5_file: str) -> ad.AnnData:
    """
    Load 10X scATAC-seq H5 file with proper format handling
    """
    print(f"📂 Loading data: {Path(h5_file).name}")
    
    # scanpy can directly read 10X H5 format
    adata = sc.read_10x_h5(h5_file, gex_only=False)
    
    print(f"   • Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
    
    # Store raw counts (guard for backends without .copy())
    # Pylance may not see .copy on backend objects; ignore if missing
    X_raw = adata.X.copy() if hasattr(adata.X, "copy") else adata.X  # type: ignore[attr-defined]
    adata.layers['counts'] = X_raw
    
    return adata


def parse_peak_names(adata: ad.AnnData, 
                     format_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Intelligently parse peak coordinates from var_names
    
    Supports formats:
    - chr1:1000-2000 (standard)
    - chr1_1000_2000 (10X format)
    - chr1-1000-2000 (alternative)
    
    Returns DataFrame with chr, start, end columns
    """
    
    peak_coords = []
    failed = []
    
    for peak_name in adata.var_names:
        # Try format 1: chr1:1000-2000 (most common)
        if ':' in peak_name and '-' in peak_name:
            try:
                chrom, coords = peak_name.split(':')
                start, end = coords.split('-')
                peak_coords.append({
                    'chr': chrom,
                    'start': int(start),
                    'end': int(end),
                    'peak_name': peak_name
                })
                continue
            except:
                pass
        
        # Try format 2: chr1_1000_2000 (10X format)
        if '_' in peak_name:
            parts = peak_name.split('_')
            if len(parts) >= 3:
                try:
                    peak_coords.append({
                        'chr': parts[0],
                        'start': int(parts[1]),
                        'end': int(parts[2]),
                        'peak_name': peak_name
                    })
                    continue
                except:
                    pass
        
        # Try format 3: chr1-1000-2000
        if peak_name.count('-') >= 2:
            match = re.match(r'^(chr[\w]+)-(\d+)-(\d+)$', peak_name)
            if match:
                peak_coords.append({
                    'chr': match.group(1),
                    'start': int(match.group(2)),
                    'end': int(match.group(3)),
                    'peak_name': peak_name
                })
                continue
        
        failed.append(peak_name)
    
    if failed:
        warnings.warn(f"Failed to parse {len(failed)}/{len(adata.var_names)} peaks")
    
    df = pd.DataFrame(peak_coords)
    
    return df


def add_peak_coordinates(adata: ad.AnnData) -> ad.AnnData:
    """
    Add chr, start, end to adata.var for downstream analysis
    """
    print("\n📍 Parsing peak coordinates")
    
    coord_df = parse_peak_names(adata)
    
    # Reindex to match adata.var
    coord_df = coord_df.set_index('peak_name')
    coord_df = coord_df.reindex(adata.var_names)
    
    # Add to adata.var
    adata.var['chr'] = coord_df['chr'].values
    adata.var['start'] = coord_df['start'].values
    adata.var['end'] = coord_df['end'].values
    adata.var['peak_width'] = adata.var['end'] - adata.var['start']
    
    # Get summary statistics
    n_chroms = adata.var['chr'].nunique()
    width_range = (int(adata.var['peak_width'].min()), int(adata.var['peak_width'].max()))
    width_median = int(adata.var['peak_width'].median())
    
    print(f"   • Parsed: {len(coord_df):,} peaks")
    print(f"   • Chromosomes: {n_chroms}")
    print(f"   • Peak width: {width_range[0]}-{width_range[1]} bp (median: {width_median} bp)")
    
    return adata


# ============================================================================
# STEP 2: Gene Annotation (Custom Implementation)
# ============================================================================

class GTFParser:
    """
    Efficient GTF parser for gene annotation
    Best practices from GENCODE/Ensembl
    """
    
    def __init__(self, gtf_file: str):
        self.gtf_file = Path(gtf_file)
        self.genes: Dict[str, list] = defaultdict(list)  # chr -> [(start, end, gene_name, gene_id, strand)]
        
    def parse(self, feature_type: str = 'gene', 
              gene_type: Optional[str] = 'protein_coding') -> Dict:
        """
        Parse GTF file and extract gene coordinates
        
        Args:
            feature_type: 'gene' or 'transcript' 
            gene_type: Filter by gene_type (e.g., 'protein_coding', 'lncRNA')
                      None = keep all
        """
        
        n_parsed = 0
        n_filtered = 0
        
        with open(self.gtf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.rstrip('\n').split('\t')
                if len(fields) < 9:
                    continue
                
                # Filter by feature type
                if fields[2] != feature_type:
                    continue
                
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                attributes = fields[8]
                
                # Parse attributes
                attr_dict = {}
                for attr in attributes.split(';'):
                    attr = attr.strip()
                    if not attr:
                        continue
                    key, value = attr.split(' ', 1)
                    attr_dict[key] = value.strip('"')
                
                # Get gene name and ID
                gene_name = attr_dict.get('gene_name', attr_dict.get('gene_id', 'unknown'))
                gene_id = attr_dict.get('gene_id', 'unknown')
                gene_biotype = attr_dict.get('gene_biotype', attr_dict.get('gene_type', 'unknown'))
                
                # Filter by gene type
                if gene_type is not None and gene_biotype != gene_type:
                    n_filtered += 1
                    continue
                
                self.genes[chrom].append({
                    'start': start,
                    'end': end,
                    'gene_name': gene_name,
                    'gene_id': gene_id,
                    'gene_type': gene_biotype,
                    'strand': strand
                })
                n_parsed += 1
        
        # Sort genes by start position for efficient lookup
        for chrom in self.genes:
            self.genes[chrom] = sorted(self.genes[chrom], key=lambda x: x['start'])
        
        return self.genes
    
    def get_gene_tss(self, upstream: int = 2000, downstream: int = 0):
        """
        Get TSS-based promoter regions
        
        Returns dict with extended TSS coordinates
        """
        tss_regions = defaultdict(list)
        
        for chrom, gene_list in self.genes.items():
            for gene in gene_list:
                if gene['strand'] == '+':
                    # TSS is at start
                    tss_start = max(0, gene['start'] - upstream)
                    tss_end = gene['start'] + downstream
                else:
                    # TSS is at end
                    tss_start = max(0, gene['end'] - downstream)
                    tss_end = gene['end'] + upstream
                
                tss_regions[chrom].append({
                    **gene,
                    'tss_start': tss_start,
                    'tss_end': tss_end
                })
        
        return tss_regions


def annotate_peaks_to_genes(
    adata: ad.AnnData,
    gtf_file: str,
    promoter_upstream: int = 2000,
    promoter_downstream: int = 500,
    gene_body: bool = True,
    distal_threshold: int = 50000,
    gene_type: Optional[str] = 'protein_coding',
    priority: Literal['promoter', 'closest', 'all'] = 'promoter'
) -> ad.AnnData:
    """
    Annotate peaks to genes using genomic context
    
    Annotation Strategy:
        1. Promoter: Peaks overlapping TSS ± window
        2. Gene body: Peaks within gene boundaries (if no promoter hit)
        3. Distal: Peaks within distance threshold (if no overlap)
        4. Intergenic: All other peaks
    
    Args:
        adata: AnnData with chr/start/end in .var (run add_peak_coordinates first)
        gtf_file: Path to GTF/GFF annotation file
        promoter_upstream: Upstream extension from TSS (bp)
        promoter_downstream: Downstream extension from TSS (bp)
        gene_body: Include gene body overlaps (True recommended)
        distal_threshold: Maximum distance for distal annotation (bp)
        gene_type: Gene biotype filter ('protein_coding', 'lncRNA', or None for all)
        priority: Multi-gene resolution strategy
            - 'promoter': Prefer promoter > gene_body > distal
            - 'closest': Assign to nearest gene
            - 'all': Keep all overlapping genes (';'-separated)
    
    Returns:
        AnnData with annotation columns added to .var:
            - gene_annotation: Annotated gene name(s)
            - annotation_type: Category (promoter/gene_body/distal/intergenic)
            - distance_to_tss: Distance to nearest TSS (bp)
    
    References:
        Signac (Stuart et al., Nat Methods 2021)
        ArchR (Granja et al., Nat Genet 2021)
    """
    
    print("\n🧬 Annotating peaks to genes")
    
    # Check required columns
    required_cols = ['chr', 'start', 'end']
    if not all(col in adata.var.columns for col in required_cols):
        raise ValueError(f"adata.var must contain {required_cols}. Run add_peak_coordinates() first.")
    
    # Parse GTF
    print(f"   • Loading GTF: {Path(gtf_file).name}")
    gtf = GTFParser(gtf_file)
    genes = gtf.parse(gene_type=gene_type)
    tss_regions = gtf.get_gene_tss(upstream=promoter_upstream, downstream=promoter_downstream)
    
    n_genes = sum(len(gene_list) for gene_list in genes.values())
    n_chroms = len(genes)
    print(f"   • Parsed: {n_genes:,} genes across {n_chroms} chromosomes")
    
    # Annotation settings
    print(f"   • Strategy: {priority}")
    print(f"   • Promoter: TSS -{promoter_upstream}/+{promoter_downstream} bp")
    print(f"   • Gene body: {'enabled' if gene_body else 'disabled'}")
    print(f"   • Distal cutoff: {distal_threshold:,} bp")
    
    # Perform annotation
    annotations = []
    annotation_types = []
    distances_to_tss = []
    
    # Iterate over peaks using explicit DataFrame reference to satisfy type checkers
    var_df = adata.var  # ensures pandas DataFrame semantics
    for idx, row in var_df.iterrows():  # type: ignore[attr-defined]
        chrom = row['chr']
        peak_start = row['start']
        peak_end = row['end']
        peak_center = (peak_start + peak_end) // 2
        
        if chrom not in genes:
            annotations.append('intergenic')
            annotation_types.append('intergenic')
            distances_to_tss.append(np.nan)
            continue
        
        # Find overlapping features
        promoter_genes = []
        gene_body_genes = []
        distal_genes = []
        min_dist_to_tss = np.inf
        
        # Check TSS/promoter overlaps
        for gene_info in tss_regions[chrom]:
            # Promoter overlap
            if not (gene_info['tss_end'] < peak_start or peak_end < gene_info['tss_start']):
                promoter_genes.append(gene_info)
                
                # Calculate distance to TSS
                if gene_info['strand'] == '+':
                    dist = abs(peak_center - gene_info['start'])
                else:
                    dist = abs(peak_center - gene_info['end'])
                min_dist_to_tss = min(min_dist_to_tss, dist)
        
        # Check gene body overlaps (if no promoter hit)
        if gene_body and not promoter_genes:
            for gene_info in genes[chrom]:
                if not (gene_info['end'] < peak_start or peak_end < gene_info['start']):
                    gene_body_genes.append(gene_info)
        
        # Check distal genes (if no overlap)
        if not promoter_genes and not gene_body_genes:
            for gene_info in tss_regions[chrom]:
                # Distance to TSS
                if gene_info['strand'] == '+':
                    tss_pos = gene_info['start']
                else:
                    tss_pos = gene_info['end']
                
                dist = abs(peak_center - tss_pos)
                
                if dist <= distal_threshold:
                    distal_genes.append((dist, gene_info))
                    min_dist_to_tss = min(min_dist_to_tss, dist)
        
        # Assign annotation based on priority
        if promoter_genes:
            if priority == 'promoter' or priority == 'closest':
                gene_name = promoter_genes[0]['gene_name']
            else:  # priority == 'all'
                gene_name = ';'.join([g['gene_name'] for g in promoter_genes])
            annotations.append(gene_name)
            annotation_types.append('promoter')
            
        elif gene_body_genes:
            if priority == 'closest':
                gene_name = min(gene_body_genes, 
                              key=lambda g: abs(peak_center - (g['start'] + g['end']) // 2))['gene_name']
            elif priority == 'all':
                gene_name = ';'.join([g['gene_name'] for g in gene_body_genes])
            else:
                gene_name = gene_body_genes[0]['gene_name']
            annotations.append(gene_name)
            annotation_types.append('gene_body')
            
        elif distal_genes:
            distal_genes.sort(key=lambda x: x[0])
            if priority == 'all':
                gene_name = ';'.join([g[1]['gene_name'] for g in distal_genes[:3]])
            else:
                gene_name = distal_genes[0][1]['gene_name']
            annotations.append(gene_name)
            annotation_types.append('distal')
            
        else:
            annotations.append('intergenic')
            annotation_types.append('intergenic')
        
        distances_to_tss.append(min_dist_to_tss if min_dist_to_tss != np.inf else np.nan)
    
    # Save to adata.var
    adata.var['gene_annotation'] = annotations
    adata.var['annotation_type'] = annotation_types
    adata.var['distance_to_tss'] = distances_to_tss
    
    # Statistics
    print("\n   📊 Annotation Summary:")
    type_counts = pd.Series(annotation_types).value_counts()
    for anno_type in ['promoter', 'gene_body', 'distal', 'intergenic']:
        if anno_type in type_counts.index:
            count = type_counts[anno_type]
            pct = count / len(annotations) * 100
            print(f"      • {anno_type.capitalize():12s}: {count:7,} peaks ({pct:5.1f}%)")
    
    n_annotated = sum(anno != 'intergenic' for anno in annotations)
    print(f"      • Total annotated: {n_annotated:,}/{len(annotations):,} peaks ({n_annotated/len(annotations)*100:.1f}%)")
    
    # Top genes
    gene_counts = pd.Series([g for g in annotations if g not in ['intergenic', 'parse_failed']]).value_counts()
    if len(gene_counts) > 0:
        print("\n   🔝 Top 5 annotated genes:")
        for gene, count in gene_counts.head(5).items():
            print(f"      • {gene}: {count} peaks")
    
    return adata

# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def annotation_pipeline(
    h5_file: str,
    gtf_file: str,
    output_h5ad: Optional[str] = None,
    # Annotation parameters
    promoter_upstream: int = 2000,
    promoter_downstream: int = 500,
    gene_body: bool = True,
    distal_threshold: int = 50000,
    gene_type: Optional[str] = 'protein_coding',
    annotation_priority: Literal['promoter', 'closest', 'all'] = 'promoter',
    # Normalization parameters
    apply_tfidf: bool = True,
    tfidf_scale_factor: float = 1e4,
    tfidf_log_tf: bool = False,
    tfidf_log_idf: bool = True,
    # HVP selection parameters
    select_hvp: bool = True,
    n_top_peaks: int = 20000,
    hvp_min_accessibility: float = 0.01,
    hvp_max_accessibility: float = 0.95,
    hvp_method: Literal['signac', 'snapatac2', 'deviance'] = 'signac',
) -> ad.AnnData:
    """
    Complete scATAC-seq peak annotation and preprocessing pipeline
    
    Workflow:
        1. Load 10X scATAC-seq H5 data
        2. Parse and validate peak coordinates
        3. Annotate peaks to genes using GTF
        4. Apply TF-IDF normalization (optional)
        5. Select highly variable peaks (optional)
        6. Save annotated AnnData object
    
    Args:
        h5_file: Path to 10X scATAC-seq H5 file (filtered_peak_bc_matrix.h5)
        gtf_file: Path to gene annotation GTF file (GENCODE/Ensembl)
        output_h5ad: Output path for annotated H5AD file (None = no save)
        
        # Peak-to-Gene Annotation
        promoter_upstream: Upstream extension from TSS (bp, default: 2000)
        promoter_downstream: Downstream extension from TSS (bp, default: 500)
        gene_body: Include gene body annotations (default: True)
        distal_threshold: Max distance for distal annotations (bp, default: 50000)
        gene_type: Gene biotype filter ('protein_coding', 'lncRNA', None=all)
        annotation_priority: Multi-gene resolution ('promoter'/'closest'/'all')
        
        # TF-IDF Normalization
        apply_tfidf: Whether to apply TF-IDF normalization (default: True)
        tfidf_scale_factor: TF-IDF scaling constant (default: 1e4)
        tfidf_log_tf: Log-transform TF component (default: False)
        tfidf_log_idf: Log-transform IDF component (default: True)
        
        # Highly Variable Peaks
        select_hvp: Whether to select highly variable peaks (default: True)
        n_top_peaks: Number of HVPs to select (default: 20000)
        hvp_min_accessibility: Minimum peak accessibility (default: 0.01)
        hvp_max_accessibility: Maximum peak accessibility (default: 0.95)
        hvp_method: Selection method ('signac'/'snapatac2'/'deviance')
    
    Returns:
        AnnData object with:
            - .X: TF-IDF normalized counts (if apply_tfidf=True)
            - .layers['counts']: Raw counts
            - .var: Peak annotations (gene_annotation, annotation_type, distance_to_tss)
            - .var['highly_variable']: Boolean mask (if select_hvp=True)
    
    Example:
        >>> adata = annotation_pipeline(
        ...     h5_file='data/filtered_peak_bc_matrix.h5',
        ...     gtf_file='annotation/gencode.v44.annotation.gtf',
        ...     output_h5ad='output/annotated_peaks.h5ad',
        ...     promoter_upstream=2000,
        ...     n_top_peaks=20000
        ... )
    
    References:
        - Signac: Stuart et al., Nat Methods 2021
        - SnapATAC2: Zhang et al., Nat Commun 2021
        - ArchR: Granja et al., Nat Genet 2021
    """
    
    print("\n" + "="*70)
    print("🚀 scATAC-seq Peak Annotation & Preprocessing Pipeline")
    print("="*70)
    
    # Step 1: Load data
    adata = load_10x_h5_data(h5_file)
    
    # Step 2: Parse peak coordinates
    adata = add_peak_coordinates(adata)
    
    # Step 3: Annotate peaks to genes
    adata = annotate_peaks_to_genes(
        adata=adata,
        gtf_file=gtf_file,
        promoter_upstream=promoter_upstream,
        promoter_downstream=promoter_downstream,
        gene_body=gene_body,
        distal_threshold=distal_threshold,
        gene_type=gene_type,
        priority=annotation_priority
    )
    
    # Step 4: TF-IDF normalization
    if apply_tfidf:
        from .utils import tfidf_normalization
        tfidf_normalization(
            adata=adata,
            scale_factor=tfidf_scale_factor,
            log_tf=tfidf_log_tf,
            log_idf=tfidf_log_idf,
            inplace=True
        )
    else:
        print("\n⏭️  Skipping TF-IDF normalization")
    
    # Step 5: Select highly variable peaks
    if select_hvp:
        from .utils import select_highly_variable_peaks 
        select_highly_variable_peaks(
            adata=adata,
            n_top_peaks=n_top_peaks,
            min_accessibility=hvp_min_accessibility,
            max_accessibility=hvp_max_accessibility,
            method=hvp_method,
            use_raw_counts=True,
            inplace=True
        )
    else:
        print("\n⏭️  Skipping HVP selection")
    
    # Step 6: Save output
    if output_h5ad:
        print("\n💾 Saving results")
        output_path = Path(output_h5ad)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_h5ad)
        print(f"   • Saved to: {output_h5ad}")
        print(f"   • File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    print("\n" + "="*70)
    print("✅ Pipeline complete!")
    print("="*70)
    print("\n📋 Final dataset:")
    print(f"   • Cells: {adata.n_obs:,}")
    print(f"   • Peaks: {adata.n_vars:,}")
    if select_hvp:
        n_hvp = adata.var['highly_variable'].sum()
        print(f"   • Highly variable peaks: {n_hvp:,}")
    print(f"   • Annotated peaks: {(adata.var['annotation_type'] != 'intergenic').sum():,}")
    print("="*70 + "\n")
    
    return adata
