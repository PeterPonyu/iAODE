import numpy as np
from scipy.sparse import issparse, csr_matrix
from typing import Literal, Tuple
import warnings

# ============================================================================
# BEST PRACTICE: TF-IDF Normalization
# ============================================================================

def tfidf_normalization(adata, 
                        scale_factor: float = 1e4,
                        log_tf: bool = False,
                        log_idf: bool = True,
                        inplace: bool = True):
    """
    TF-IDF normalization following Signac/SnapATAC2 best practices
    
    RECOMMENDED SETTINGS:
    - scale_factor = 1e4 (standard for most datasets)
    - log_tf = False (standard TF)
    - log_idf = True (log(1 + n_cells/n_cells_with_peak))
    
    Alternative for very large/sparse datasets:
    - scale_factor = 1e6
    - log_tf = True (log(1 + TF))
    
    Args:
        adata: AnnData object with raw counts
        scale_factor: Scale factor (1e4 standard, 1e6 for large datasets)
        log_tf: Whether to log-transform TF (False is standard)
        log_idf: Whether to log-transform IDF (True is standard)
        inplace: Modify adata in place
    
    Returns:
        adata with TF-IDF normalized data (or None if inplace=True)
        
    References:
        - Signac: https://github.com/stuart-lab/signac
        - SnapATAC2: https://github.com/kaizhang/SnapATAC2
    """
    
    if not inplace:
        adata = adata.copy()
    
    print(f"Applying TF-IDF normalization (scale={scale_factor:.0e})...")
    
    # Get counts matrix
    if issparse(adata.X):
        X = adata.X.copy()
    else:
        X = csr_matrix(adata.X.copy())
    
    # ========================================================================
    # Term Frequency (TF)
    # ========================================================================
    
    # Calculate total counts per cell
    cell_sums = np.array(X.sum(axis=1)).flatten()
    cell_sums[cell_sums == 0] = 1  # Avoid division by zero
    
    # Normalize: peak_count / total_counts_per_cell
    tf = X.multiply(1.0 / cell_sums[:, None])
    
    # Optional: log-transform TF (log(1 + TF))
    if log_tf:
        tf.data = np.log1p(tf.data)
        print("  Using log-transformed TF: log(1 + TF)")
    
    # ========================================================================
    # Inverse Document Frequency (IDF)
    # ========================================================================
    
    n_cells = adata.n_obs
    
    # Count cells where each peak is accessible (count > 0)
    n_cells_per_peak = np.array((X > 0).sum(axis=0)).flatten()
    n_cells_per_peak[n_cells_per_peak == 0] = 1  # Avoid division by zero
    
    if log_idf:
        # Standard: log(1 + n_cells / n_cells_with_peak)
        idf = np.log1p(n_cells / n_cells_per_peak)
    else:
        # Alternative: n_cells / n_cells_with_peak
        idf = n_cells / n_cells_per_peak
    
    # ========================================================================
    # TF-IDF = TF * IDF * scale_factor
    # ========================================================================
    
    tfidf = tf.multiply(idf)
    tfidf = tfidf.multiply(scale_factor)
    
    adata.X = tfidf.tocsr()
    
    # Store normalization parameters
    adata.uns['tfidf_params'] = {
        'scale_factor': scale_factor,
        'log_tf': log_tf,
        'log_idf': log_idf
    }
    
    print(f"  TF-IDF complete. Value range: [{tfidf.min():.2e}, {tfidf.max():.2e}]")
    
    return adata if not inplace else None


# ============================================================================
# BEST PRACTICE: Highly Variable Peak Selection
# ============================================================================

def select_highly_variable_peaks(
    adata,
    n_top_peaks: int = 20000,
    min_accessibility: float = 0.01,  # Peak must be accessible in ≥1% cells
    max_accessibility: float = 0.95,  # Filter ubiquitous peaks
    method: Literal['signac', 'snapatac2', 'deviance'] = 'signac',
    use_raw_counts: bool = True,      # Use raw counts for selection
    inplace: bool = True
):
    """
    Select highly variable peaks using best practices
    
    RECOMMENDED METHOD: 'signac' (variance-based after filtering)
    
    Methods:
        - 'signac': Variance on TF-IDF after accessibility filtering (DEFAULT)
        - 'snapatac2': Variance-to-mean ratio with accessibility weighting
        - 'deviance': Binomial deviance on raw counts
    
    Args:
        adata: AnnData object (should have 'counts' layer if use_raw_counts=True)
        n_top_peaks: Number of peaks to select
        min_accessibility: Minimum fraction of cells with peak > 0
        max_accessibility: Maximum fraction of cells (filter ubiquitous peaks)
        method: Selection method
        use_raw_counts: Use raw counts layer for deviance method
        inplace: Modify adata in place
    
    Returns:
        adata with 'highly_variable' in adata.var (or None if inplace=True)
        
    References:
        - Signac: Stuart et al., Nat Methods 2021
        - SnapATAC2: Zhang et al., Nat Commun 2021
    """
    
    if not inplace:
        adata = adata.copy()
    
    print(f"\n{'='*70}")
    print(f"Selecting highly variable peaks: method='{method}'")
    print(f"{'='*70}")
    
    # ========================================================================
    # Step 1: Calculate peak accessibility
    # ========================================================================
    
    if use_raw_counts and 'counts' in adata.layers:
        X = adata.layers['counts']
    else:
        X = adata.X
    
    if issparse(X):
        n_cells_per_peak = np.array((X > 0).sum(axis=0)).flatten()
    else:
        n_cells_per_peak = np.sum(X > 0, axis=0)
    
    accessibility = n_cells_per_peak / adata.n_obs
    adata.var['accessibility'] = accessibility
    
    print(f"Accessibility range: [{accessibility.min():.4f}, {accessibility.max():.4f}]")
    print(f"Median accessibility: {np.median(accessibility):.4f}")
    
    # ========================================================================
    # Step 2: Filter by accessibility
    # ========================================================================
    
    accessibility_mask = (accessibility >= min_accessibility) & (accessibility <= max_accessibility)
    n_filtered = (~accessibility_mask).sum()
    
    print(f"\nFiltering peaks by accessibility ({min_accessibility:.1%} - {max_accessibility:.1%}):")
    print(f"  Removed {n_filtered:,} peaks ({n_filtered/adata.n_vars*100:.1f}%)")
    print(f"  Retained {accessibility_mask.sum():,} peaks")
    
    # ========================================================================
    # Step 3: Calculate variability scores
    # ========================================================================
    
    if method == 'signac':
        # --------------------------------------------------------------------
        # Signac Method: Variance on TF-IDF normalized data
        # --------------------------------------------------------------------
        print("\nMethod: Signac (variance on TF-IDF)")
        
        X_norm = adata.X
        
        if issparse(X_norm):
            # Efficient variance calculation for sparse matrices
            mean = np.array(X_norm.mean(axis=0)).flatten()
            mean_sq = np.array(X_norm.power(2).mean(axis=0)).flatten()
            variance = mean_sq - mean ** 2
        else:
            variance = np.var(X_norm, axis=0)
        
        # Set variance to -inf for filtered peaks
        variance[~accessibility_mask] = -np.inf
        
        adata.var['variance'] = variance
        adata.var['highly_variable_rank'] = np.argsort(np.argsort(-variance))
        
        score = variance
        score_name = 'variance'
        
    elif method == 'snapatac2':
        # --------------------------------------------------------------------
        # SnapATAC2 Method: Variance-to-mean ratio weighted by accessibility
        # --------------------------------------------------------------------
        print("\nMethod: SnapATAC2 (VMR × accessibility)")
        
        X_norm = adata.X
        
        if issparse(X_norm):
            mean = np.array(X_norm.mean(axis=0)).flatten()
            mean_sq = np.array(X_norm.power(2).mean(axis=0)).flatten()
            variance = mean_sq - mean ** 2
        else:
            mean = np.mean(X_norm, axis=0)
            variance = np.var(X_norm, axis=0)
        
        # Variance-to-mean ratio (VMR)
        vmr = np.zeros_like(variance)
        valid_mask = mean > 0
        vmr[valid_mask] = variance[valid_mask] / mean[valid_mask]
        
        # Weight by accessibility
        score = vmr * accessibility
        
        # Set score to -inf for filtered peaks
        score[~accessibility_mask] = -np.inf
        
        adata.var['vmr'] = vmr
        adata.var['vmr_weighted'] = score
        adata.var['highly_variable_rank'] = np.argsort(np.argsort(-score))
        
        score_name = 'vmr_weighted'
        
    elif method == 'deviance':
        # --------------------------------------------------------------------
        # Deviance Method: Binomial deviance on raw counts
        # --------------------------------------------------------------------
        print("\nMethod: Binomial deviance (on raw counts)")
        
        # Use raw counts
        if use_raw_counts and 'counts' in adata.layers:
            X_counts = adata.layers['counts']
        else:
            warnings.warn("Using adata.X for deviance (should use raw counts)")
            X_counts = adata.X
        
        if issparse(X_counts):
            X_binary = (X_counts > 0).astype(float).toarray()
        else:
            X_binary = (X_counts > 0).astype(float)
        
        # Mean accessibility (already calculated)
        p = accessibility.copy()
        p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
        
        # Binomial deviance (vectorized)
        deviance = -2 * (
            X_binary * np.log(p)[None, :] + 
            (1 - X_binary) * np.log(1 - p)[None, :]
        ).sum(axis=0)
        
        # Set deviance to -inf for filtered peaks
        deviance[~accessibility_mask] = -np.inf
        
        adata.var['deviance'] = deviance
        adata.var['highly_variable_rank'] = np.argsort(np.argsort(-deviance))
        
        score = deviance
        score_name = 'deviance'
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # ========================================================================
    # Step 4: Select top peaks (FIX: Avoid chained assignment warning)
    # ========================================================================
    
    n_top_peaks = min(n_top_peaks, accessibility_mask.sum())
    
    # Get indices of top peaks
    top_idx = np.argsort(-score)[:n_top_peaks]
    
    # Mark highly variable peaks - FIXED: Use .loc with proper indexing
    adata.var['highly_variable'] = False
    adata.var.loc[adata.var.index[top_idx], 'highly_variable'] = True
    
    print(f"\nSelected {n_top_peaks:,} highly variable peaks")
    print(f"  {score_name} range: [{score[top_idx].min():.2e}, {score[top_idx].max():.2e}]")
    print(f"  Mean accessibility of selected peaks: {accessibility[top_idx].mean():.4f}")
    print(f"{'='*70}\n")
    
    return adata if not inplace else None



def get_dataset_category(n_cells: int) -> Tuple[str, int, int]:
    """
    Categorize dataset by size and return configuration
    
    Args:
        n_cells: Number of cells in dataset
        
    Returns:
        Tuple of (category_name, subsample_size, n_hvp)
    """
    if n_cells < 5000:
        return 'tiny', None, None  # Skip processing
    elif n_cells < 10000:
        return 'small', 5000, 20000
    elif n_cells < 20000:
        return 'medium', 10000, 20000
    else:
        return 'large', 20000, 20000