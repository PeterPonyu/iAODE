#!/usr/bin/env python3
"""
Generate simulated test data for iAODE Training UI testing
Creates small synthetic scRNA-seq and scATAC-seq datasets
"""

import numpy as np
import pandas as pd
from scipy import sparse
import os

def create_test_scrna_data():
    """Create a small synthetic scRNA-seq dataset"""
    print("Generating scRNA-seq test data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Small dataset: 500 cells x 2000 genes
    n_cells = 500
    n_genes = 2000
    
    # Generate count matrix with negative binomial distribution
    # Simulate 3 cell types with different expression patterns
    cell_type_labels = []
    counts_list = []
    
    for i in range(3):
        n_cells_type = n_cells // 3
        if i == 2:  # Last group gets remainder
            n_cells_type = n_cells - (n_cells // 3) * 2
        
        # Different mean expression for each cell type
        mu = np.random.gamma(2, 2, n_genes) * (i + 1)
        
        # Generate counts for this cell type
        counts = np.random.negative_binomial(
            n=5, 
            p=5 / (5 + mu), 
            size=(n_cells_type, n_genes)
        )
        counts_list.append(counts)
        cell_type_labels.extend([f'CellType{i+1}'] * n_cells_type)
    
    # Combine all cell types
    X = np.vstack(counts_list)
    
    # Create AnnData object
    try:
        import anndata
        
        # Cell metadata
        obs = pd.DataFrame({
            'cell_type': cell_type_labels,
            'n_counts': X.sum(axis=1),
            'n_genes': (X > 0).sum(axis=1)
        })
        obs.index = [f'Cell_{i}' for i in range(n_cells)]
        
        # Gene metadata
        var = pd.DataFrame({
            'gene_name': [f'Gene_{i}' for i in range(n_genes)],
            'highly_variable': np.random.choice([True, False], n_genes, p=[0.2, 0.8])
        })
        var.index = var['gene_name']
        
        # Create AnnData
        adata = anndata.AnnData(
            X=sparse.csr_matrix(X),
            obs=obs,
            var=var
        )
        
        # Add counts layer
        adata.layers['counts'] = adata.X.copy()
        
        # Save
        os.makedirs('test_data', exist_ok=True)
        adata.write_h5ad('test_data/test_scrna.h5ad')
        print(f"✓ Created test_data/test_scrna.h5ad")
        print(f"  Shape: {adata.shape}")
        print(f"  Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
        
        return adata
        
    except ImportError:
        print("Error: anndata not installed. Install with: pip install anndata")
        return None

def create_test_scatac_data():
    """Create a small synthetic scATAC-seq dataset"""
    print("\nGenerating scATAC-seq test data...")
    
    # Set random seed
    np.random.seed(43)
    
    # Small dataset: 300 cells x 5000 peaks
    n_cells = 300
    n_peaks = 5000
    
    # Generate sparse binary/count matrix (peaks are more sparse than genes)
    cell_type_labels = []
    counts_list = []
    
    for i in range(2):  # 2 cell types for ATAC
        n_cells_type = n_cells // 2
        if i == 1:
            n_cells_type = n_cells - n_cells // 2
        
        # ATAC data is typically more sparse
        # ~5-10% of peaks accessible per cell
        prob_accessible = 0.05 + i * 0.03
        
        counts = np.zeros((n_cells_type, n_peaks))
        for j in range(n_cells_type):
            # Randomly select accessible peaks
            accessible = np.random.random(n_peaks) < prob_accessible
            # Count accessibility (0-10 reads per accessible peak)
            counts[j, accessible] = np.random.poisson(2, accessible.sum())
        
        counts_list.append(counts)
        cell_type_labels.extend([f'CellType{i+1}'] * n_cells_type)
    
    # Combine
    X = np.vstack(counts_list)
    
    try:
        import anndata
        
        # Cell metadata
        obs = pd.DataFrame({
            'cell_type': cell_type_labels,
            'n_counts': X.sum(axis=1),
            'n_peaks': (X > 0).sum(axis=1)
        })
        obs.index = [f'Cell_{i}' for i in range(n_cells)]
        
        # Peak metadata (chr:start-end format)
        var = pd.DataFrame({
            'peak_name': [f'chr1:{i*1000}-{i*1000+500}' for i in range(n_peaks)],
            'highly_variable': False  # Will be computed during preprocessing
        })
        var.index = var['peak_name']
        
        # Create AnnData
        adata = anndata.AnnData(
            X=sparse.csr_matrix(X),
            obs=obs,
            var=var
        )
        
        # Add counts layer
        adata.layers['counts'] = adata.X.copy()
        
        # Save
        adata.write_h5ad('test_data/test_scatac.h5ad')
        print(f"✓ Created test_data/test_scatac.h5ad")
        print(f"  Shape: {adata.shape}")
        print(f"  Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
        
        return adata
        
    except ImportError:
        print("Error: anndata not installed")
        return None

def print_summary():
    """Print summary of generated test data"""
    print("\n" + "="*60)
    print("Test Data Generation Complete!")
    print("="*60)
    print("\nGenerated files in test_data/:")
    print("  1. test_scrna.h5ad  - scRNA-seq dataset (500 cells × 2000 genes)")
    print("  2. test_scatac.h5ad - scATAC-seq dataset (300 cells × 5000 peaks)")
    print("\nYou can now:")
    print("  1. Start the training UI: ./start_training_ui.sh")
    print("  2. Upload test_data/test_scrna.h5ad or test_data/test_scatac.h5ad")
    print("  3. Configure training parameters and start training")
    print("\nNote: These are small synthetic datasets for testing purposes.")
    print("      Training should complete in 1-2 minutes.")
    print("="*60)

def main():
    print("="*60)
    print("iAODE Test Data Generator")
    print("="*60)
    print()
    
    # Create test data
    scrna_data = create_test_scrna_data()
    scatac_data = create_test_scatac_data()
    
    if scrna_data is not None and scatac_data is not None:
        print_summary()
        return 0
    else:
        print("\nError: Could not generate test data.")
        print("Install required packages: pip install anndata scanpy")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
