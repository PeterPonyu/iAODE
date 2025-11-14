
"""
Complete HOMER Motif Enrichment Analysis for scATAC-seq
Includes installation, genome setup, and analysis for human and mouse
"""

import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Literal
import anndata as ad
from scipy.sparse import issparse
import warnings
import re

# ============================================================================
# PART 1: HOMER INSTALLATION AND SETUP
# ============================================================================

class HOMERSetup:
    """
    Helper class for HOMER installation and genome configuration
    """
    
    @staticmethod
    def check_homer_installation() -> bool:
        """
        Check if HOMER is installed and accessible
        """
        try:
            result = subprocess.run(
                ['findMotifsGenome.pl'],
                capture_output=True,
                text=True
            )
            print("✅ HOMER is installed and accessible")
            return True
        except FileNotFoundError:
            print("❌ HOMER not found in PATH")
            return False
    
    
    @staticmethod
    def print_installation_instructions():
        """
        Print detailed HOMER installation instructions
        """
        
        instructions = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    HOMER INSTALLATION INSTRUCTIONS                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

METHOD 1: Install via Conda (RECOMMENDED - Easiest)
────────────────────────────────────────────────────────────────────────────
# Create conda environment (optional but recommended)
conda create -n homer_env python=3.9
conda activate homer_env

# Install HOMER
conda install -c bioconda homer

# Verify installation
findMotifsGenome.pl

────────────────────────────────────────────────────────────────────────────

METHOD 2: Manual Installation from Source
────────────────────────────────────────────────────────────────────────────
# Download HOMER
cd ~
mkdir homer
cd homer
wget http://homer.ucsd.edu/homer/configureHomer.pl
perl configureHomer.pl -install

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=$PATH:~/homer/bin/

# Reload shell
source ~/.bashrc  # or source ~/.zshrc

# Verify installation
findMotifsGenome.pl

────────────────────────────────────────────────────────────────────────────

METHOD 3: Using Docker (For reproducibility)
────────────────────────────────────────────────────────────────────────────
# Pull HOMER docker image
docker pull biocontainers/homer:v4.11_cv2

# Run HOMER in container
docker run -v $(pwd):/data biocontainers/homer:v4.11_cv2 findMotifsGenome.pl

────────────────────────────────────────────────────────────────────────────

NEXT STEPS: Install Genomes (see install_genome() method)

════════════════════════════════════════════════════════════════════════════
"""
        print(instructions)
    
    
    @staticmethod
    def install_genome(organism: Literal['human', 'mouse'],
                       assembly: Optional[str] = None,
                       force: bool = False) -> bool:
        """
        Install genome for HOMER
        
        Args:
            organism: 'human' or 'mouse'
            assembly: Specific assembly version (None = latest)
                     Human: 'hg38' (default), 'hg19'
                     Mouse: 'mm10' (default), 'mm39', 'mm9'
            force: Reinstall even if already present
        
        Returns:
            True if successful
        """
        
        # Default assemblies
        default_assemblies = {
            'human': 'hg38',
            'mouse': 'mm10'
        }
        
        if assembly is None:
            assembly = default_assemblies[organism]
        
        print(f"\n{'='*70}")
        print(f"Installing HOMER genome: {assembly} ({organism})")
        print(f"{'='*70}\n")
        
        # Check if already installed
        if not force:
            check_cmd = ['configureHomer.pl', '-list']
            try:
                result = subprocess.run(check_cmd, capture_output=True, text=True)
                if assembly in result.stdout:
                    print(f"✅ {assembly} is already installed")
                    return True
            except:
                pass
        
        # Install genome
        install_cmd = ['configureHomer.pl', '-install', assembly]
        
        print(f"Running: {' '.join(install_cmd)}")
        print("This may take 5-15 minutes depending on your internet speed...")
        
        try:
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                print(f"\n✅ Successfully installed {assembly}")
                return True
            else:
                print(f"\n❌ Installation failed:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"\n❌ Installation timed out (>30 min)")
            return False
        except Exception as e:
            print(f"\n❌ Installation error: {e}")
            return False
    
    
    @staticmethod
    def list_installed_genomes():
        """
        List all installed HOMER genomes
        """
        try:
            result = subprocess.run(
                ['configureHomer.pl', '-list'],
                capture_output=True,
                text=True
            )
            
            print("\n" + "="*70)
            print("INSTALLED HOMER GENOMES")
            print("="*70)
            
            # Parse installed genomes
            lines = result.stdout.split('\n')
            installed = []
            
            for line in lines:
                if 'installed' in line.lower():
                    # Extract genome name
                    match = re.search(r'(\w+)\s+', line)
                    if match:
                        installed.append(match.group(1))
            
            if installed:
                print("\n✅ Installed genomes:")
                for genome in installed:
                    print(f"   • {genome}")
            else:
                print("\n⚠️  No genomes installed")
                print("\nRecommended installations:")
                print("   • Human: hg38 (GRCh38)")
                print("   • Mouse: mm10 (GRCm38)")
            
            print("\nTo install a genome:")
            print("   configureHomer.pl -install <genome>")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"Error listing genomes: {e}")
    
    
    @staticmethod
    def download_genome_fasta(organism: Literal['human', 'mouse'],
                             assembly: Optional[str] = None,
                             output_dir: str = './genomes') -> Path:
        """
        Download genome FASTA file (needed for sequence extraction)
        
        Args:
            organism: 'human' or 'mouse'
            assembly: Genome assembly version
            output_dir: Where to save FASTA file
        
        Returns:
            Path to downloaded FASTA file
        """
        
        default_assemblies = {
            'human': 'hg38',
            'mouse': 'mm10'
        }
        
        if assembly is None:
            assembly = default_assemblies[organism]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fasta_file = output_path / f"{assembly}.fa"
        
        if fasta_file.exists():
            print(f"✅ Genome FASTA already exists: {fasta_file}")
            return fasta_file
        
        print(f"\n{'='*70}")
        print(f"Downloading genome FASTA: {assembly}")
        print(f"{'='*70}\n")
        print("⚠️  This is a large file (>1 GB) and may take 10-30 minutes")
        
        # URLs for genome downloads
        urls = {
            'hg38': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',
            'hg19': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz',
            'mm10': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz',
            'mm39': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz',
            'mm9': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm9/bigZips/mm9.fa.gz',
        }
        
        if assembly not in urls:
            raise ValueError(f"Unknown assembly: {assembly}")
        
        url = urls[assembly]
        gz_file = output_path / f"{assembly}.fa.gz"
        
        print(f"Downloading from: {url}")
        print(f"Saving to: {fasta_file}")
        
        # Download
        download_cmd = ['wget', '-O', str(gz_file), url]
        
        try:
            subprocess.run(download_cmd, check=True)
            print("\n✅ Download complete")
            
            # Decompress
            print("Decompressing...")
            subprocess.run(['gunzip', str(gz_file)], check=True)
            
            print(f"✅ Genome FASTA ready: {fasta_file}")
            return fasta_file
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\nAlternative: Download manually from UCSC Genome Browser")
            print(f"   {url}")
            raise


# ============================================================================
# PART 2: PREPARE INPUT FILES FOR HOMER
# ============================================================================

class HOMERInputPreparation:
    """
    Prepare peak files and sequences for HOMER analysis
    """
    
    @staticmethod
    def adata_to_bed(adata: ad.AnnData,
                     peak_list: List[str],
                     output_file: str,
                     score_column: Optional[str] = None) -> Path:
        """
        Convert AnnData peaks to BED file for HOMER
        
        HOMER BED format requirements:
        1. chr
        2. start
        3. end
        4. name (unique ID)
        5. score (optional)
        6. strand (+ or -)
        
        Args:
            adata: AnnData with peak annotations
            peak_list: List of peak names to export
            output_file: Output BED file path
            score_column: Column in adata.var to use as score (optional)
        
        Returns:
            Path to BED file
        """
        
        print(f"\n📝 Creating BED file for HOMER")
        print(f"   • Peaks: {len(peak_list):,}")
        
        # Extract peak information
        peak_df = adata.var.loc[peak_list, ['chr', 'start', 'end']].copy()
        
        # Add peak names as 4th column
        peak_df['name'] = peak_list
        
        # Add score (5th column)
        if score_column and score_column in adata.var.columns:
            peak_df['score'] = adata.var.loc[peak_list, score_column]
        else:
            peak_df['score'] = 1000  # Default score
        
        # Add strand (6th column) - HOMER requires this
        peak_df['strand'] = '+'  # Default to + strand
        
        # Reorder columns
        bed_df = peak_df[['chr', 'start', 'end', 'name', 'score', 'strand']]
        
        # Convert to integers
        bed_df['start'] = bed_df['start'].astype(int)
        bed_df['end'] = bed_df['end'].astype(int)
        
        # Save to file (no header, tab-separated)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        bed_df.to_csv(output_path, sep='\t', header=False, index=False)
        
        print(f"   • Saved: {output_path}")
        print(f"   • File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Show preview
        print(f"\n   Preview (first 3 lines):")
        with open(output_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    print(f"   {line.rstrip()}")
        
        return output_path
    
    
    @staticmethod
    def create_background_bed(adata: ad.AnnData,
                             foreground_peaks: List[str],
                             output_file: str,
                             strategy: Literal['all', 'random', 'matched'] = 'random',
                             n_background: Optional[int] = None,
                             match_annotation: bool = False) -> Path:
        """
        Create background peak set for HOMER
        
        Args:
            adata: AnnData with all peaks
            foreground_peaks: Peaks used as foreground
            output_file: Output BED file
            strategy: How to select background
                - 'all': All peaks except foreground
                - 'random': Random subset of non-foreground peaks
                - 'matched': Match annotation type distribution
            n_background: Number of background peaks (for random/matched)
            match_annotation: Match annotation type distribution
        
        Returns:
            Path to background BED file
        """
        
        print(f"\n📝 Creating background BED file")
        print(f"   • Strategy: {strategy}")
        
        # Get all peaks except foreground
        all_peaks = set(adata.var_names)
        foreground_set = set(foreground_peaks)
        background_candidates = list(all_peaks - foreground_set)
        
        print(f"   • Foreground peaks: {len(foreground_peaks):,}")
        print(f"   • Available background: {len(background_candidates):,}")
        
        # Select background based on strategy
        if strategy == 'all':
            background_peaks = background_candidates
            
        elif strategy == 'random':
            if n_background is None:
                n_background = min(len(background_candidates), len(foreground_peaks) * 2)
            
            n_background = min(n_background, len(background_candidates))
            background_peaks = np.random.choice(
                background_candidates,
                size=n_background,
                replace=False
            ).tolist()
            
        elif strategy == 'matched':
            # Match annotation type distribution
            if 'annotation_type' not in adata.var.columns:
                warnings.warn("annotation_type not found, falling back to random")
                strategy = 'random'
                return HOMERInputPreparation.create_background_bed(
                    adata, foreground_peaks, output_file, 'random', n_background
                )
            
            # Get foreground annotation distribution
            fg_annot = adata.var.loc[foreground_peaks, 'annotation_type'].value_counts()
            
            if n_background is None:
                n_background = len(foreground_peaks) * 2
            
            # Sample from each annotation type proportionally
            background_peaks = []
            bg_candidates_df = adata.var.loc[background_candidates]
            
            for annot_type, count in fg_annot.items():
                n_sample = int(count / len(foreground_peaks) * n_background)
                candidates = bg_candidates_df[
                    bg_candidates_df['annotation_type'] == annot_type
                ].index.tolist()
                
                if len(candidates) > 0:
                    n_sample = min(n_sample, len(candidates))
                    sampled = np.random.choice(candidates, size=n_sample, replace=False)
                    background_peaks.extend(sampled)
            
            print(f"   • Matched annotation distribution")
        
        print(f"   • Selected background peaks: {len(background_peaks):,}")
        
        # Create BED file
        return HOMERInputPreparation.adata_to_bed(
            adata,
            background_peaks,
            output_file
        )


# ============================================================================
# PART 3: RUN HOMER MOTIF ENRICHMENT
# ============================================================================

class HOMERMotifEnrichment:
    """
    Run HOMER motif enrichment analysis
    """
    
    @staticmethod
    def run_homer(
        peak_bed: str,
        genome: str,
        output_dir: str,
        # Core parameters
        size: int = 200,
        mask: bool = True,
        # Background
        background_bed: Optional[str] = None,
        # Motif parameters
        motif_length: str = '8,10,12',
        num_motifs: int = 25,
        num_mismatches: int = 2,
        # Performance
        n_cpus: int = 4,
        # Advanced options
        use_hypergeometric: bool = False,
        norevopp: bool = False,
        nomotif: bool = False,  # Skip de novo, only known motifs
        # Additional
        extra_args: Optional[List[str]] = None
    ) -> Path:
        """
        Run HOMER findMotifsGenome.pl
        
        Args:
            peak_bed: Path to BED file with peaks
            genome: Genome assembly (e.g., 'hg38', 'mm10')
            output_dir: Output directory
            size: Region size for motif finding (bp)
                 200 = standard for TF peaks
                 500-1000 = for histone marks
                 Use negative values for offset (e.g., '-300,100' for upstream bias)
            mask: Use repeat-masked genome
            background_bed: Custom background peaks (optional)
            motif_length: Comma-separated motif lengths (e.g., '8,10,12')
            num_motifs: Number of motifs to find
            num_mismatches: Mismatches in optimization (increase for longer motifs)
            n_cpus: Number of CPUs
            use_hypergeometric: Use hypergeometric (good for small background)
            norevopp: Don't search reverse strand
            nomotif: Skip de novo discovery (faster, only known motifs)
            extra_args: Additional command-line arguments
        
        Returns:
            Path to output directory
        """
        
        print("\n" + "="*70)
        print("🧬 RUNNING HOMER MOTIF ENRICHMENT")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            'findMotifsGenome.pl',
            peak_bed,
            genome,
            str(output_path),
            '-size', str(size),
            '-len', motif_length,
            '-S', str(num_motifs),
            '-mis', str(num_mismatches),
            '-p', str(n_cpus)
        ]
        
        # Add mask
        if mask:
            cmd.append('-mask')
        
        # Add background
        if background_bed:
            cmd.extend(['-bg', background_bed])
            print(f"   • Using custom background: {background_bed}")
        
        # Add hypergeometric
        if use_hypergeometric:
            cmd.append('-h')
        
        # Add norevopp
        if norevopp:
            cmd.append('-norevopp')
        
        # Skip de novo
        if nomotif:
            cmd.append('-nomotif')
            print(f"   • Skipping de novo motif discovery (known motifs only)")
        
        # Extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        print(f"\n📋 HOMER Configuration:")
        print(f"   • Input peaks: {peak_bed}")
        print(f"   • Genome: {genome}")
        print(f"   • Output: {output_path}")
        print(f"   • Region size: {size} bp")
        print(f"   • Motif lengths: {motif_length}")
        print(f"   • Number of motifs: {num_motifs}")
        print(f"   • CPUs: {n_cpus}")
        print(f"   • Repeat masking: {mask}")
        
        print(f"\n💻 Command:")
        print(f"   {' '.join(cmd)}")
        
        print(f"\n⏳ Running HOMER (this may take 5-30 minutes)...")
        print("="*70 + "\n")
        
        # Run HOMER
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print("\n" + "="*70)
                print("✅ HOMER COMPLETED SUCCESSFULLY")
                print("="*70)
                
                # Print summary from stdout
                if result.stdout:
                    print("\nHOMER Output:")
                    # Print last 20 lines of output
                    lines = result.stdout.split('\n')
                    for line in lines[-20:]:
                        if line.strip():
                            print(f"   {line}")
                
                return output_path
            else:
                print("\n" + "="*70)
                print("❌ HOMER FAILED")
                print("="*70)
                print("\nError output:")
                print(result.stderr)
                raise RuntimeError("HOMER execution failed")
                
        except subprocess.TimeoutExpired:
            print("\n❌ HOMER timed out (>1 hour)")
            raise
        except Exception as e:
            print(f"\n❌ HOMER error: {e}")
            raise
    
    
    @staticmethod
    def parse_homer_results(output_dir: str,
                           result_type: Literal['known', 'denovo'] = 'known',
                           top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Parse HOMER output files
        
        Args:
            output_dir: HOMER output directory
            result_type: 'known' or 'denovo'
            top_n: Return only top N results
        
        Returns:
            DataFrame with motif enrichment results
        """
        
        output_path = Path(output_dir)
        
        if result_type == 'known':
            result_file = output_path / 'knownResults.txt'
        else:
            result_file = output_path / 'homerResults.txt'
        
        if not result_file.exists():
            raise FileNotFoundError(f"HOMER results not found: {result_file}")
        
        print(f"\n📊 Parsing HOMER results: {result_file}")
        
        # Read results
        results = pd.read_csv(result_file, sep='\t')
        
        print(f"   • Found {len(results)} motifs")
        
        if top_n:
            results = results.head(top_n)
            print(f"   • Showing top {len(results)} motifs")
        
        return results
    
    
    @staticmethod
    def summarize_homer_output(output_dir: str) -> Dict:
        """
        Summarize HOMER output and provide file locations
        
        Returns:
            Dictionary with summary information
        """
        
        output_path = Path(output_dir)
        
        print("\n" + "="*70)
        print("📁 HOMER OUTPUT SUMMARY")
        print("="*70)
        
        summary = {
            'output_dir': str(output_path),
            'files': {}
        }
        
        # Key output files
        key_files = {
            'knownResults.html': 'Known motif enrichment (open in browser)',
            'knownResults.txt': 'Known motif enrichment (tab-delimited)',
            'homerResults.html': 'De novo motif discovery (open in browser)',
            'homerResults.txt': 'De novo motif discovery (tab-delimited)',
            'homerMotifs.all.motifs': 'All de novo motifs (HOMER format)',
            'motifFindingParameters.txt': 'Command used',
            'seq.autonorm.tsv': 'Autonormalization statistics'
        }
        
        print("\n📄 Output Files:")
        for filename, description in key_files.items():
            filepath = output_path / filename
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                print(f"   ✅ {filename}")
                print(f"      {description}")
                print(f"      Size: {size_kb:.1f} KB")
                summary['files'][filename] = str(filepath)
            else:
                print(f"   ❌ {filename} (not found)")
        
        # Check for HTML results
        html_known = output_path / 'knownResults.html'
        html_denovo = output_path / 'homerResults.html'
        
        if html_known.exists():
            print(f"\n🌐 View Known Motif Results:")
            print(f"   Open in browser: {html_known.absolute()}")
            summary['known_motifs_html'] = str(html_known.absolute())
        
        if html_denovo.exists():
            print(f"\n🌐 View De Novo Motif Results:")
            print(f"   Open in browser: {html_denovo.absolute()}")
            summary['denovo_motifs_html'] = str(html_denovo.absolute())
        
        # Parse top results
        try:
            known_results = HOMERMotifEnrichment.parse_homer_results(
                output_dir, 'known', top_n=10
            )
            print(f"\n🏆 Top 10 Known Motifs:")
            print(known_results[['Motif Name', 'P-value', '% of Target Sequences', 
                                '% of Background Sequences']].to_string(index=False))
            summary['top_known_motifs'] = known_results
        except:
            print("\n⚠️  Could not parse known motif results")
        
        print("\n" + "="*70 + "\n")
        
        return summary


# ============================================================================
# PART 4: COMPLETE PIPELINE
# ============================================================================

def complete_homer_pipeline(
    adata: ad.AnnData,
    peak_list: List[str],
    organism: Literal['human', 'mouse'],
    output_dir: str = 'homer_analysis',
    # Genome options
    genome_assembly: Optional[str] = None,
    install_genome: bool = True,
    # Background options
    use_background: bool = True,
    background_strategy: Literal['all', 'random', 'matched'] = 'random',
    n_background: Optional[int] = None,
    # HOMER parameters
    region_size: int = 200,
    motif_lengths: str = '8,10,12',
    num_motifs: int = 25,
    mask_repeats: bool = True,
    n_cpus: int = 4,
    skip_denovo: bool = False,
    # Filtering
    filter_peaks: bool = True,
    min_accessibility: float = 0.01,
    annotation_types: Optional[List[str]] = None
) -> Dict:
    """
    Complete HOMER motif enrichment pipeline
    
    Takes AnnData object and peak list → Runs HOMER → Returns results
    
    Args:
        adata: AnnData with peak annotations
        peak_list: List of peak names to analyze
        organism: 'human' or 'mouse'
        output_dir: Output directory
        genome_assembly: Genome version (None = use default)
        install_genome: Automatically install genome if not present
        use_background: Create custom background peaks
        background_strategy: How to select background
        n_background: Number of background peaks
        region_size: Size of region for motif finding
        motif_lengths: Motif lengths to search
        num_motifs: Number of motifs to find
        mask_repeats: Use repeat-masked genome
        n_cpus: Number of CPUs
        skip_denovo: Skip de novo discovery (faster)
        filter_peaks: Apply quality filters to peak list
        min_accessibility: Minimum peak accessibility
        annotation_types: Filter by annotation types
    
    Returns:
        Dictionary with results and file paths
    
    Example:
        >>> results = complete_homer_pipeline(
        ...     adata=adata,
        ...     peak_list=monocyte_peaks,
        ...     organism='human',
        ...     output_dir='monocyte_motifs',
        ...     region_size=200,
        ...     annotation_types=['distal', 'promoter']
        ... )
        >>> # View results in browser
        >>> import webbrowser
        >>> webbrowser.open(results['known_motifs_html'])
    """
    
    print("\n" + "="*70)
    print("🚀 COMPLETE HOMER MOTIF ENRICHMENT PIPELINE")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Check HOMER installation
    print("\n📦 Step 1: Checking HOMER installation")
    if not HOMERSetup.check_homer_installation():
        print("\n⚠️  HOMER is not installed")
        HOMERSetup.print_installation_instructions()
        raise RuntimeError("Please install HOMER first")
    
    # Step 2: Setup genome
    print("\n🧬 Step 2: Setting up genome")
    
    default_assemblies = {'human': 'hg38', 'mouse': 'mm10'}
    if genome_assembly is None:
        genome_assembly = default_assemblies[organism]
    
    print(f"   • Organism: {organism}")
    print(f"   • Assembly: {genome_assembly}")
    
    if install_genome:
        HOMERSetup.install_genome(organism, genome_assembly)
    
    # Step 3: Filter peaks (optional)
    if filter_peaks:
        print("\n🔍 Step 3: Filtering peak list")
        print(f"   • Input peaks: {len(peak_list):,}")
        
        peak_df = adata.var.loc[peak_list].copy()
        
        # Accessibility filter
        if 'accessibility' in peak_df.columns:
            before = len(peak_df)
            peak_df = peak_df[peak_df['accessibility'] >= min_accessibility]
            print(f"   • After accessibility filter (>={min_accessibility}): {len(peak_df):,}")
        
        # Annotation type filter
        if annotation_types and 'annotation_type' in peak_df.columns:
            peak_df = peak_df[peak_df['annotation_type'].isin(annotation_types)]
            print(f"   • After annotation filter ({annotation_types}): {len(peak_df):,}")
        
        peak_list = peak_df.index.tolist()
        print(f"   • Final peak count: {len(peak_list):,}")
    
    if len(peak_list) < 50:
        warnings.warn("Very few peaks (<50). Results may not be reliable.")
    
    # Step 4: Create input files
    print("\n📝 Step 4: Creating input files")
    
    peak_bed = output_path / 'target_peaks.bed'
    HOMERInputPreparation.adata_to_bed(adata, peak_list, str(peak_bed))
    
    background_bed = None
    if use_background:
        background_bed = output_path / 'background_peaks.bed'
        HOMERInputPreparation.create_background_bed(
            adata,
            peak_list,
            str(background_bed),
            strategy=background_strategy,
            n_background=n_background
        )
    
    # Step 5: Run HOMER
    print("\n🔬 Step 5: Running HOMER motif enrichment")
    
    homer_output = output_path / 'homer_results'
    
    HOMERMotifEnrichment.run_homer(
        peak_bed=str(peak_bed),
        genome=genome_assembly,
        output_dir=str(homer_output),
        size=region_size,
        mask=mask_repeats,
        background_bed=str(background_bed) if background_bed else None,
        motif_length=motif_lengths,
        num_motifs=num_motifs,
        n_cpus=n_cpus,
        nomotif=skip_denovo
    )
    
    # Step 6: Parse and summarize results
    print("\n📊 Step 6: Summarizing results")
    
    summary = HOMERMotifEnrichment.summarize_homer_output(str(homer_output))
    
    # Add pipeline info
    summary['pipeline_info'] = {
        'organism': organism,
        'genome': genome_assembly,
        'n_peaks': len(peak_list),
        'region_size': region_size,
        'used_background': use_background,
        'output_dir': str(output_path)
    }
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📂 All results saved to: {output_path.absolute()}")
    print(f"\n🌐 View results in browser:")
    if 'known_motifs_html' in summary:
        print(f"   Known motifs: {summary['known_motifs_html']}")
    if 'denovo_motifs_html' in summary:
        print(f"   De novo motifs: {summary['denovo_motifs_html']}")
    print("="*70 + "\n")
    
    return summary


# ============================================================================
# HELPER: HOMER LOGO EXPLANATION
# ============================================================================

def explain_homer_logos():
    """
    Explain what HOMER sequence logos represent
    """
    
    explanation = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    HOMER SEQUENCE LOGO EXPLANATION                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

WHAT IS A SEQUENCE LOGO?
────────────────────────────────────────────────────────────────────────────
A sequence logo is a graphical representation of a DNA motif (binding site).

Each position in the motif shows:
  • HEIGHT of letters = Information content (bits)
    - Taller = more conserved position
    - Shorter = more variable position
  
  • SIZE of each letter = Relative frequency
    - Larger A = More A's at this position
    - Smaller A = Fewer A's at this position

EXAMPLE LOGO INTERPRETATION:
────────────────────────────────────────────────────────────────────────────
       
       A
    GGGGAA
   TGGGGAAC     ← This represents a motif
  TTGGGGAACT
  
Position 1: Mostly T (with some variation)
Position 2-5: Highly conserved GGGG (tall letters = strong conservation)
Position 6-7: Mostly AA (medium conservation)
Position 8-9: Variable C/T (short letters = weak conservation)

HOMER OUTPUTS TWO TYPES OF LOGOS:
────────────────────────────────────────────────────────────────────────────

1️⃣  KNOWN MOTIF LOGOS (knownResults.html)
   • Shows established TF binding motifs from databases (JASPAR, etc.)
   • Each logo represents a known transcription factor
   • Example: CTCF logo, SP1 logo, NF-κB logo
   
   📊 The logo shows: "What does this TF typically bind to?"
   
   These are PRE-DEFINED motifs that HOMER checks for enrichment

2️⃣  DE NOVO MOTIF LOGOS (homerResults.html)
   • Shows NEW motifs discovered by HOMER in YOUR peaks
   • HOMER finds these by analyzing the sequences
   • May match known TFs, or may be novel patterns
   
   📊 The logo shows: "What sequence patterns are enriched in your peaks?"
   
   These are DISCOVERED from your data

HOW HOMER CREATES LOGOS:
────────────────────────────────────────────────────────────────────────────

For your peak set:
1. HOMER extracts DNA sequences from each peak region
2. Scans for enriched sequence patterns
3. Aligns matching sequences
4. Counts nucleotide frequency at each position
5. Calculates information content
6. Generates logo

The logo represents:
  ✅ The CONSENSUS sequence that TFs bind to
  ✅ Based on ACTUAL sequences from YOUR peaks
  ✅ NOT just general A/T/C/G frequency (that would be meaningless)

LOGO IN HTML OUTPUT:
────────────────────────────────────────────────────────────────────────────

When you open knownResults.html or homerResults.html, you'll see:

┌─────────────────────────────────────────────────────────────────────────┐
│ Rank | Motif Name | Logo | P-value | % Target | % Background | ...     │
├─────────────────────────────────────────────────────────────────────────┤
│  1   | CTCF       | [Logo Image] | 1e-150 | 45.2% | 12.3% | ...        │
│  2   | PU.1       | [Logo Image] | 1e-89  | 32.1% | 8.9%  | ...        │
│  ... | ...        | ...          | ...    | ...   | ...   | ...        │
└─────────────────────────────────────────────────────────────────────────┘

Each logo is:
  • A PNG image showing the motif
  • Clickable to see more details
  • Represents the binding preference of that TF

ADDITIONAL HOMER OUTPUTS:
────────────────────────────────────────────────────────────────────────────

📁 homerResults/ directory contains:
   • motif1.motif, motif2.motif, ... (HOMER motif format files)
   • motif1.logo.png, motif2.logo.png, ... (Logo images)
   • motif1.info.html (Detailed information per motif)

You can use these .motif files for:
   • Finding motif instances in specific peaks
   • Scanning new sequences
   • Comparing to other motifs

KEY TAKEAWAY:
────────────────────────────────────────────────────────────────────────────
HOMER logos show the DNA binding preferences (motifs) of transcription 
factors that are ENRICHED in your peak set compared to background.

The height and size of letters indicate how strongly conserved each position
is - NOT just random nucleotide frequencies!

════════════════════════════════════════════════════════════════════════════
"""
    
    print(explanation)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    
    # Print logo explanation
    explain_homer_logos()
    
    print("\n" + "="*70)
    print("EXAMPLE USAGE")
    print("="*70 + "\n")
    
    example_code = """
# Load annotated AnnData
import scanpy as sc
adata = sc.read_h5ad('annotated_peaks.h5ad')

# Define peak list (e.g., from latent component analysis)
monocyte_peaks = ['chr1:1000-1500', 'chr2:2000-2500', ...]  # Your peak list

# Run complete HOMER pipeline
from homer_motif_enrichment import complete_homer_pipeline

results = complete_homer_pipeline(
    adata=adata,
    peak_list=monocyte_peaks,
    organism='human',  # or 'mouse'
    output_dir='monocyte_homer_analysis',
    
    # Optional: filtering
    filter_peaks=True,
    annotation_types=['distal', 'promoter'],  # Focus on regulatory regions
    min_accessibility=0.02,
    
    # Optional: HOMER parameters
    region_size=200,  # 200bp around peak center (standard for TFs)
    motif_lengths='8,10,12',  # Search for 8bp, 10bp, and 12bp motifs
    use_background=True,  # Use custom background (recommended)
    n_cpus=8,
    skip_denovo=False  # Set True to only check known motifs (faster)
)

# View results in browser
import webbrowser
webbrowser.open(results['known_motifs_html'])

# Access top motifs
top_motifs = results['top_known_motifs']
print(top_motifs[['Motif Name', 'P-value']].head(10))
"""
    
    print(example_code)
