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

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================

def load_10x_h5_data(h5_file: str) -> ad.AnnData:
    """
    Load 10X scATAC-seq H5 file with proper format handling
    """
    print(f"📂 Loading {h5_file}...")
    
    # scanpy can directly read 10X H5 format
    adata = sc.read_10x_h5(h5_file, gex_only=False)
    
    print(f"  ✓ Loaded {adata.n_obs} cells × {adata.n_vars} peaks")
    print(f"  Peak format: {adata.var_names[0]}")
    
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()
    
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
    
    print("\n🔍 Parsing peak coordinates...")
    
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
        print(f"  ⚠️  Failed to parse {len(failed)}/{len(adata.var_names)} peaks")
        print(f"  Examples: {failed[:3]}")
    
    df = pd.DataFrame(peak_coords)
    print(f"  ✓ Parsed {len(df)} peaks successfully")
    print(f"  Chromosomes: {sorted(df['chr'].unique())[:10]}...")
    
    return df


def add_peak_coordinates(adata: ad.AnnData) -> ad.AnnData:
    """
    Add chr, start, end to adata.var for downstream analysis
    """
    coord_df = parse_peak_names(adata)
    
    # Reindex to match adata.var
    coord_df = coord_df.set_index('peak_name')
    coord_df = coord_df.reindex(adata.var_names)
    
    # Add to adata.var
    adata.var['chr'] = coord_df['chr'].values
    adata.var['start'] = coord_df['start'].values
    adata.var['end'] = coord_df['end'].values
    adata.var['peak_width'] = adata.var['end'] - adata.var['start']
    
    print(f"  ✓ Added chr, start, end to adata.var")
    print(f"  Peak width range: {adata.var['peak_width'].min()}-{adata.var['peak_width'].max()}bp")
    
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
        self.genes = defaultdict(list)  # chr -> [(start, end, gene_name, gene_id, strand)]
        
    def parse(self, feature_type: str = 'gene', 
              gene_type: Optional[str] = 'protein_coding') -> Dict:
        """
        Parse GTF file and extract gene coordinates
        
        Args:
            feature_type: 'gene' or 'transcript' 
            gene_type: Filter by gene_type (e.g., 'protein_coding', 'lncRNA')
                      None = keep all
        """
        
        print(f"\n📖 Parsing GTF: {self.gtf_file}")
        print(f"  Feature type: {feature_type}")
        print(f"  Gene type filter: {gene_type}")
        
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
        
        print(f"  ✓ Parsed {n_parsed:,} genes across {len(self.genes)} chromosomes")
        if gene_type:
            print(f"  Filtered out {n_filtered:,} non-{gene_type} genes")
        
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
    Annotate peaks to genes with multiple strategies
    
    Best practices:
    - Promoter: TSS ± 2kb (default)
    - Gene body: entire gene span
    - Distal: up to 50kb from gene
    
    Priority modes:
    - 'promoter': Prefer promoter > gene body > distal
    - 'closest': Assign to nearest gene
    - 'all': Keep all overlapping genes (separated by ';')
    
    Args:
        adata: AnnData with chr/start/end in .var
        gtf_file: Path to GTF/GFF file
        promoter_upstream: bp upstream of TSS
        promoter_downstream: bp downstream of TSS
        gene_body: Include gene body overlaps
        distal_threshold: Max distance for distal annotation
        gene_type: Filter to specific gene types (None = all)
        priority: How to handle multiple genes per peak
    """
    
    print("\n" + "="*70)
    print("🧬 Peak-to-Gene Annotation")
    print("="*70)
    
    # Check required columns
    required_cols = ['chr', 'start', 'end']
    if not all(col in adata.var.columns for col in required_cols):
        raise ValueError(f"adata.var must contain {required_cols}. Run add_peak_coordinates() first.")
    
    # Parse GTF
    gtf = GTFParser(gtf_file)
    genes = gtf.parse(gene_type=gene_type)
    tss_regions = gtf.get_gene_tss(upstream=promoter_upstream, downstream=promoter_downstream)
    
    # Annotation
    print(f"\n🔗 Annotating peaks...")
    print(f"  Strategy: {priority}")
    print(f"  Promoter: TSS ±{promoter_upstream}/{promoter_downstream}bp")
    print(f"  Gene body: {gene_body}")
    print(f"  Distal cutoff: {distal_threshold}bp")
    
    annotations = []
    annotation_types = []
    distances_to_tss = []
    
    for idx, row in adata.var.iterrows():
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
                # Take first promoter gene (or closest if multiple)
                gene_name = promoter_genes[0]['gene_name']
            else:  # priority == 'all'
                gene_name = ';'.join([g['gene_name'] for g in promoter_genes])
            annotations.append(gene_name)
            annotation_types.append('promoter')
            
        elif gene_body_genes:
            if priority == 'closest':
                # Take gene closest to center
                gene_name = min(gene_body_genes, 
                              key=lambda g: abs(peak_center - (g['start'] + g['end']) // 2))['gene_name']
            elif priority == 'all':
                gene_name = ';'.join([g['gene_name'] for g in gene_body_genes])
            else:
                gene_name = gene_body_genes[0]['gene_name']
            annotations.append(gene_name)
            annotation_types.append('gene_body')
            
        elif distal_genes:
            # Sort by distance
            distal_genes.sort(key=lambda x: x[0])
            if priority == 'all':
                gene_name = ';'.join([g[1]['gene_name'] for g in distal_genes[:3]])  # Top 3
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
    print(f"\n📊 Annotation Statistics:")
    type_counts = pd.Series(annotation_types).value_counts()
    for anno_type, count in type_counts.items():
        pct = count / len(annotations) * 100
        print(f"  {anno_type:12s}: {count:6,} ({pct:5.1f}%)")
    
    n_annotated = sum(anno != 'intergenic' for anno in annotations)
    print(f"\n  Total annotated: {n_annotated:,}/{len(annotations):,} ({n_annotated/len(annotations)*100:.1f}%)")
    
    # Top annotated genes
    gene_counts = pd.Series([g for g in annotations if g not in ['intergenic', 'parse_failed']]).value_counts()
    print(f"\n🔝 Top annotated genes:")
    for gene, count in gene_counts.head(10).items():
        print(f"  {gene}: {count} peaks")
    
    print("="*70 + "\n")
    
    return adata

# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def annotation_pipeline(
    h5_file: str,
    gtf_file: str,
    output_h5ad: Optional[str] = None,
    **annotation_kwargs
) -> ad.AnnData:
    """
    Complete pipeline from H5 to annotated AnnData
    
    Args:
        h5_file: Path to 10X H5 file
        gtf_file: Path to GTF annotation file
        output_h5ad: Save output (optional)
        **annotation_kwargs: Pass to annotate_peaks_to_genes()
    """
    
    print("\n" + "="*70)
    print("🚀 scATAC-seq Peak Annotation Pipeline")
    print("="*70)
    
    # Step 1: Load data
    adata = load_10x_h5_data(h5_file)
    
    # Step 2: Parse peak coordinates
    adata = add_peak_coordinates(adata)
    
    # Step 3: Annotate peaks to genes
    adata = annotate_peaks_to_genes(adata, gtf_file, **annotation_kwargs)
    
    # Step 4: Save
    if output_h5ad:
        print(f"\n💾 Saving to {output_h5ad}...")
        adata.write_h5ad(output_h5ad)
        print(f"  ✓ Saved successfully")
    
    print("\n✅ Pipeline complete!")
    print("="*70 + "\n")
    
    return adata
