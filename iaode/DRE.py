import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import time
import warnings
from typing import Dict, Tuple


class DimensionalityReductionEvaluator:
    """
    Dimensionality reduction quality evaluator.
    
    Core metrics:
    - distance_correlation: global distance correlation (global structure preservation)
    - Q_global: global quality
    - Q_local: local quality
    
    Features:
    - Efficient, vectorized computation
    - Focus on key evaluation metrics
    - Complementary to the single-cell latent space evaluation framework
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            verbose: whether to print detailed information
        """
        self.verbose = verbose
        
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _validate_inputs(self, X_high: np.ndarray, X_low: np.ndarray, k: int):
        """Validate input parameters."""
        if not isinstance(X_high, np.ndarray) or not isinstance(X_low, np.ndarray):
            raise TypeError("Input data must be NumPy arrays.")
        
        if X_high.shape[0] != X_low.shape[0]:
            raise ValueError(
                f"High- and low-dimensional data must have the same number of samples: "
                f"{X_high.shape[0]} vs {X_low.shape[0]}"
            )
        
        if k >= X_high.shape[0]:
            raise ValueError(f"k ({k}) must be smaller than the number of samples ({X_high.shape[0]}).")
            
        if X_high.ndim != 2 or X_low.ndim != 2:
            raise ValueError("Input data must be 2D arrays.")
    
    # ==================== 1. Distance correlation ====================
    
    def distance_correlation_score(self, X_high: np.ndarray, X_low: np.ndarray) -> float:
        """
        Compute distance correlation (Spearman correlation) between distances
        in the high- and low-dimensional spaces.
        
        Args:
            X_high: high-dimensional data
            X_low: low-dimensional data
            
        Returns:
            float: distance correlation score (values near 1 indicate good global structure preservation)
        """
        try:
            self._log("Computing distance matrices...")
            
            D_high = pairwise_distances(X_high)
            D_low = pairwise_distances(X_low)
            
            distance_corr, _ = spearmanr(D_high.ravel(), D_low.ravel())
            
            return distance_corr if not np.isnan(distance_corr) else 0.0
            
        except Exception as e:
            warnings.warn(f"Error computing distance correlation: {e}")
            return 0.0
    
    # ==================== 2. Ranking matrix ====================
    
    def get_ranking_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the ranking matrix (optimized).
        
        Args:
            distance_matrix: pairwise distance matrix
            
        Returns:
            np.ndarray: ranking matrix
        """
        try:
            n = distance_matrix.shape[0]
            
            # Get sorted indices row-wise
            sorted_indices = np.argsort(distance_matrix, axis=1)
            
            ranking_matrix = np.zeros((n, n), dtype=np.int32)
            
            # Assign ranks row-wise
            for i in range(n):
                ranking_matrix[i, sorted_indices[i]] = np.arange(n)
            
            # Exclude self: set diagonal to 0 and shift others by -1
            mask = np.eye(n, dtype=bool)
            ranking_matrix[~mask] = ranking_matrix[~mask] - 1
            ranking_matrix[mask] = 0
            
            return ranking_matrix
            
        except Exception as e:
            warnings.warn(f"Error computing ranking matrix: {e}")
            n = distance_matrix.shape[0]
            return np.zeros((n, n), dtype=np.int32)
    
    # ==================== 3. Co-ranking matrix ====================
    
    def get_coranking_matrix(self, rank_high: np.ndarray, rank_low: np.ndarray) -> np.ndarray:
        """
        Compute the co-ranking matrix (optimized).
        
        Args:
            rank_high: ranking matrix in high-dimensional space
            rank_low: ranking matrix in low-dimensional space
            
        Returns:
            np.ndarray: co-ranking matrix
        """
        try:
            n = rank_high.shape[0]
            corank = np.zeros((n - 1, n - 1), dtype=np.int32)
            
            mask = (rank_high > 0) & (rank_low > 0)
            valid_high = rank_high[mask] - 1  # convert to 0-based
            valid_low = rank_low[mask] - 1
            
            valid_mask = (valid_high < n - 1) & (valid_low < n - 1)
            valid_high = valid_high[valid_mask]
            valid_low = valid_low[valid_mask]
            
            np.add.at(corank, (valid_high, valid_low), 1)
            
            return corank
            
        except Exception as e:
            warnings.warn(f"Error computing co-ranking matrix: {e}")
            n = rank_high.shape[0]
            return np.zeros((n - 1, n - 1), dtype=np.int32)
    
    # ==================== 4. Q metrics ====================
    
    def compute_qnx_series(self, corank: np.ndarray) -> np.ndarray:
        """
        Compute the Q_NX series.
        
        Args:
            corank: co-ranking matrix
            
        Returns:
            np.ndarray: sequence of Q_NX values
        """
        try:
            n = corank.shape[0] + 1
            qnx_values = []
            Qnx_cum = 0
            
            for K in range(1, n - 1):
                if K - 1 < corank.shape[0]:
                    intrusions = np.sum(corank[:K, K - 1]) if K - 1 < corank.shape[1] else 0
                    extrusions = np.sum(corank[K - 1, :K]) if K - 1 < corank.shape[0] else 0
                    diagonal = corank[K - 1, K - 1] if K - 1 < min(corank.shape) else 0
                    
                    Qnx_increment = intrusions + extrusions - diagonal
                    Qnx_cum += Qnx_increment
                    
                    qnx_normalized = Qnx_cum / (K * n)
                    qnx_values.append(qnx_normalized)
            
            return np.array(qnx_values)
            
        except Exception as e:
            warnings.warn(f"Error computing Q_NX series: {e}")
            return np.array([0.0])
    
    def get_q_local_global(self, qnx_values: np.ndarray) -> Tuple[float, float, int]:
        """
        Compute scalar local and global quality metrics.
        
        Args:
            qnx_values: Q_NX value sequence
            
        Returns:
            tuple: (Q_local, Q_global, K_max)
        """
        try:
            if len(qnx_values) == 0:
                return 0.0, 0.0, 1
            
            lcmc = np.copy(qnx_values)
            N = len(qnx_values)
            
            for j in range(N):
                lcmc[j] = lcmc[j] - j / N
            
            K_max = np.argmax(lcmc) + 1
            
            if K_max > 0:
                Q_local = np.mean(qnx_values[:K_max])
            else:
                Q_local = qnx_values[0] if len(qnx_values) > 0 else 0.0
                
            if K_max < len(qnx_values):
                Q_global = np.mean(qnx_values[K_max:])
            else:
                Q_global = qnx_values[-1] if len(qnx_values) > 0 else 0.0
            
            return Q_local, Q_global, K_max
            
        except Exception as e:
            warnings.warn(f"Error computing Q metrics: {e}")
            return 0.0, 0.0, 1
    
    # ==================== 5. Comprehensive evaluation ====================
    
    def comprehensive_evaluation(
        self, X_high: np.ndarray, X_low: np.ndarray, k: int = 10
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of dimensionality reduction quality.
        
        Args:
            X_high: high-dimensional data, shape = (n_samples, n_features_high)
            X_low: low-dimensional data, shape = (n_samples, n_features_low)
            k: number of neighbors considered
            
        Returns:
            dict: dictionary with core evaluation metrics
        """
        self._validate_inputs(X_high, X_low, k)
        
        self._log(f"Starting dimensionality reduction evaluation (n_samples={X_high.shape[0]}, k={k})...")
        
        results: Dict[str, float] = {}
        
        # 1. Distance correlation
        self._log("Computing distance correlation...")
        results['distance_correlation'] = self.distance_correlation_score(X_high, X_low)
        
        # 2. Ranking matrices
        self._log("Computing ranking matrices...")
        D_high = pairwise_distances(X_high)
        D_low = pairwise_distances(X_low)
        
        rank_high = self.get_ranking_matrix(D_high)
        rank_low = self.get_ranking_matrix(D_low)
        
        # 3. Co-ranking matrix
        self._log("Computing co-ranking matrix...")
        corank = self.get_coranking_matrix(rank_high, rank_low)
        
        # 4. Q metrics
        self._log("Computing Q metrics...")
        qnx_values = self.compute_qnx_series(corank)
        Q_local, Q_global, K_max = self.get_q_local_global(qnx_values)
        
        results['Q_local'] = Q_local
        results['Q_global'] = Q_global
        results['K_max'] = K_max
        
        overall_quality = np.mean([
            results['distance_correlation'],
            results['Q_local'],
            results['Q_global'],
        ])
        results['overall_quality'] = overall_quality
        
        if self.verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, float]):
        """Print a formatted summary of evaluation results."""
        
        print("\n" + "=" * 60)
        print("            Dimensionality Reduction Quality")
        print("=" * 60)
        
        print("\n[Core Metrics]")
        print(f"  Distance correlation: {results['distance_correlation']:.4f} ★")
        print("    └─ Values near 1 indicate good global structure preservation")
        
        print(f"\n  Local quality (Q_local): {results['Q_local']:.4f} ★")
        print("    └─ Values near 1 indicate good local structure preservation")
        
        print(f"\n  Global quality (Q_global): {results['Q_global']:.4f} ★")
        print("    └─ Values near 1 indicate good global structure preservation")
        
        print("\n[Auxiliary]")
        print(f"  Local–global split point (K_max): {results['K_max']}")
        
        overall_quality = results['overall_quality']
        
        print("\n[Overall Assessment]")
        print(f"  Mean quality score: {overall_quality:.4f}")
        
        if overall_quality >= 0.8:
            quality_level = "Excellent"
        elif overall_quality >= 0.6:
            quality_level = "Good"
        elif overall_quality >= 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Needs improvement"
            
        print(f"  Quality level: {quality_level}")
        
        print("=" * 60)
    
    def compare_methods(self, method_results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], k: int = 10) -> pd.DataFrame:
        """
        Compare multiple dimensionality reduction methods.
        
        Args:
            method_results_dict: mapping {method_name: (X_high, X_low)}
            k: number of neighbors considered
            
        Returns:
            DataFrame: comparison table of methods
        """
        comparison_results = []
        
        for method_name, (X_high, X_low) in method_results_dict.items():
            self._log(f"\nEvaluating method: {method_name}")
            
            original_verbose = self.verbose
            self.verbose = False
            
            results = self.comprehensive_evaluation(X_high, X_low, k)
            
            self.verbose = original_verbose
            
            overall_quality = np.mean([
                results['distance_correlation'],
                results['Q_local'],
                results['Q_global'],
            ])
            
            comparison_results.append({
                'Method': method_name,
                'Distance_Correlation': results['distance_correlation'],
                'Q_Local': results['Q_local'],
                'Q_Global': results['Q_global'],
                'Overall_Quality': overall_quality,
            })
        
        df = pd.DataFrame(comparison_results)
        df = df.sort_values('Overall_Quality', ascending=False)
        
        if self.verbose:
            self._print_comparison_table(df)
        
        return df
    
    def _print_comparison_table(self, df: pd.DataFrame):
        """Print a formatted comparison table of methods."""
        
        print(f"\n{'=' * 90}")
        print("                    Dimensionality Reduction Method Comparison")
        print('=' * 90)
        
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print(df.to_string(index=False))
        
        print(
            f"\nBest method: {df.iloc[0]['Method']} "
            f"(Overall score: {df.iloc[0]['Overall_Quality']:.4f})"
        )
        
        print('=' * 90)


# ==================== Convenience functions ====================

def evaluate_dimensionality_reduction(
    X_high: np.ndarray, X_low: np.ndarray, k: int = 10, verbose: bool = True
) -> Dict[str, float]:
    """
    Convenience function to evaluate the quality of a dimensionality reduction.
    
    Args:
        X_high: high-dimensional data
        X_low: low-dimensional data
        k: number of neighbors considered
        verbose: whether to print detailed output
        
    Returns:
        dict: evaluation results
    """
    evaluator = DimensionalityReductionEvaluator(verbose=verbose)
    return evaluator.comprehensive_evaluation(X_high, X_low, k)


def compare_dimensionality_reduction_methods(
    method_results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    k: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to compare different dimensionality reduction methods.
    
    Args:
        method_results_dict: mapping {method_name: (X_high, X_low)}
        k: number of neighbors considered
        verbose: whether to print detailed output
        
    Returns:
        DataFrame: comparison results
    """
    evaluator = DimensionalityReductionEvaluator(verbose=verbose)
    return evaluator.compare_methods(method_results_dict, k)