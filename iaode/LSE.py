import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import entropy, skew, kurtosis
from scipy.linalg import svd, norm
import warnings


class SingleCellLatentSpaceEvaluator:
    """
    Latent space quality evaluator tailored to single-cell data.
    
    Especially suitable for:
    - Single-cell trajectory data (development, differentiation, etc.)
    - Single-cell steady-state population data
    - Time-series single-cell data
    
    Key properties:
    - Metric interpretations adapted for trajectory data
    - Low isotropy = good (strong directionality)
    - Low participation ratio = good (information concentration)
    - High spectral decay = good (dimensional efficiency)
    """
    
    def __init__(self, data_type="trajectory", verbose=True):
        """
        Initialize the evaluator.
        
        Args:
            data_type: "trajectory" or "steady_state"
            verbose: whether to print detailed logs
        """
        self.data_type = data_type
        self.verbose = verbose
        
        # Adjust preferences according to data type
        if data_type == "trajectory":
            self.isotropy_preference = "low"       # trajectories prefer low isotropy
            self.participation_preference = "low"  # trajectories prefer low participation ratio
        else:  # steady_state
            self.isotropy_preference = "high"      # steady state prefers high isotropy
            self.participation_preference = "high" # steady state prefers high participation ratio
    
    def _log(self, message):
        if self.verbose:
            print(message)
    
    # ==================== 1. Refined manifold dimensionality consistency ====================
    
    def manifold_dimensionality_score_v2(self, latent_space, 
                                         variance_thresholds=[0.8, 0.9, 0.95],
                                         use_multiple_methods=True):
        """
        Refined manifold dimensionality consistency metric.
        Fixes the issue in the original version where all methods yielded the same score.
        
        Args:
            latent_space: latent coordinates
            variance_thresholds: list of variance thresholds
            use_multiple_methods: whether to combine multiple methods
            
        Returns:
            float: dimensional efficiency score (0–1)
        """
        try:
            if latent_space.shape[1] == 1:
                return 1.0
            
            # Center data
            centered_data = latent_space - np.mean(latent_space, axis=0)
            
            # PCA
            pca = PCA().fit(centered_data)
            explained_variance_ratio = pca.explained_variance_ratio_
            explained_variance = pca.explained_variance_
            
            dimension_scores = []
            
            # Method 1: multi-threshold dimensional efficiency
            for threshold in variance_thresholds:
                cumsum = np.cumsum(explained_variance_ratio)
                effective_dims = np.where(cumsum >= threshold)[0]
                
                if len(effective_dims) > 0:
                    effective_dim = effective_dims[0] + 1
                    # Fewer dimensions to reach the threshold = higher efficiency
                    efficiency = 1.0 - (effective_dim - 1) / (latent_space.shape[1] - 1)
                    dimension_scores.append(efficiency)
            
            # Method 2: Kaiser criterion efficiency
            normalized_eigenvalues = explained_variance / np.mean(explained_variance)
            kaiser_dim = np.sum(normalized_eigenvalues > 1.0)
            kaiser_efficiency = 1.0 - (kaiser_dim - 1) / (latent_space.shape[1] - 1)
            
            # Method 3: elbow method
            if len(explained_variance) > 2:
                ratios = explained_variance[:-1] / explained_variance[1:]
                elbow_dim = np.argmax(ratios) + 1
                elbow_efficiency = 1.0 - (elbow_dim - 1) / (latent_space.shape[1] - 1)
            else:
                elbow_efficiency = 1.0
            
            # Method 4: spectral decay
            if len(explained_variance) > 1:
                log_eigenvals = np.log(explained_variance + 1e-10)
                x = np.arange(len(log_eigenvals))
                
                if len(x) > 1:
                    slope = np.polyfit(x, log_eigenvals, 1)[0]
                    # More negative slope => faster decay => stronger concentration
                    decay_score = 1.0 / (1.0 + np.exp(slope))
                else:
                    decay_score = 0.5
            else:
                decay_score = 0.5
            
            # Aggregate score
            if use_multiple_methods:
                all_scores = dimension_scores + [kaiser_efficiency, elbow_efficiency, decay_score]
                final_score = np.mean([s for s in all_scores if s is not None])
            else:
                final_score = np.mean(dimension_scores) if dimension_scores else 0.5
            
            return np.clip(final_score, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"Error computing manifold dimensionality consistency: {e}")
            return 0.5
    
    # ==================== 2. Intrinsic efficiency metrics ====================
    
    def spectral_decay_rate(self, latent_space):
        """Spectral decay rate — higher means better concentration across dimensions."""
        try:
            centered_data = latent_space - np.mean(latent_space, axis=0)
            U, s, Vt = svd(centered_data, full_matrices=False)
            eigenvalues = s**2 / (len(latent_space) - 1)
            
            if len(eigenvalues) < 2:
                return 1.0
            
            # Fit exponential decay in log space
            log_eigenvals = np.log(eigenvalues + 1e-10)
            x = np.arange(len(log_eigenvals))
            
            slope, _ = np.polyfit(x, log_eigenvals, 1)
            normalized_decay = 1.0 / (1.0 + np.exp(slope))
            
            # Concentration of the first eigenvalue
            concentration = eigenvalues[0] / np.sum(eigenvalues)
            
            # Combined score
            spectral_score = 0.6 * normalized_decay + 0.4 * concentration
            
            return np.clip(spectral_score, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"Error computing spectral decay rate: {e}")
            return 0.5
    
    def participation_ratio_score(self, latent_space):
        """
        Participation ratio score.
        
        For trajectory data: lower is better (information more concentrated).
        For steady-state data: higher is better (more even spread).
        """
        try:
            centered_data = latent_space - np.mean(latent_space, axis=0)
            cov_matrix = np.cov(centered_data.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            if len(eigenvalues) == 0:
                return 0.0
            
            # Participation ratio
            sum_eigenvals = np.sum(eigenvalues)
            sum_eigenvals_squared = np.sum(eigenvalues**2)
            
            if sum_eigenvals_squared > 0:
                participation_ratio = sum_eigenvals**2 / sum_eigenvals_squared
                max_participation = len(eigenvalues)
                normalized_pr = participation_ratio / max_participation
            else:
                normalized_pr = 0.0
            
            # Adjust according to data type
            if self.participation_preference == "low":
                # trajectories prefer low participation
                score = 1.0 - normalized_pr
            else:
                # steady state prefers high participation
                score = normalized_pr
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"Error computing participation ratio: {e}")
            return 0.5
    

    def isotropy_anisotropy_score(self, latent_space):
        """
        Isotropy/anisotropy score (enhanced version).
        
        For trajectory data: low isotropy (high anisotropy) is better (strong directionality).
        For steady-state data: high isotropy is better (uniform spreading).
        
        Enhancements:
        - Log transforms for higher sensitivity and reduced saturation
        - Multiple complementary measures to improve discriminability
        - Adaptive sensitivity via nonlinear mappings
        """
        try:
            centered_data = latent_space - np.mean(latent_space, axis=0)
            cov_matrix = np.cov(centered_data.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            
            if len(eigenvalues) < 2:
                return 1.0
            
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Method 1: log-ellipticity (mitigates saturation)
            log_ellipticity = np.log(eigenvalues[0]) - np.log(eigenvalues[-1] + 1e-12)
            enhanced_ellipticity = np.tanh(log_ellipticity / 4.0)
            
            # Method 2: multi-level condition numbers
            condition_ratios = []
            for i in range(len(eigenvalues) - 1):
                ratio = eigenvalues[i] / (eigenvalues[i + 1] + 1e-12)
                condition_ratios.append(np.log(ratio))
            
            mean_log_condition = np.mean(condition_ratios)
            enhanced_condition = np.tanh(mean_log_condition / 2.0)
            
            # Method 3: variance of adjacent ratios
            ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-12)
            ratio_variance = np.var(np.log(ratios))
            ratio_anisotropy = np.tanh(ratio_variance)
            
            # Method 4: entropy-based anisotropy
            eigenval_probs = eigenvalues / np.sum(eigenvalues)
            eigenval_entropy = -np.sum(eigenval_probs * np.log(eigenval_probs + 1e-12))
            max_entropy = np.log(len(eigenvalues))
            entropy_isotropy = eigenval_entropy / max_entropy if max_entropy > 0 else 0
            entropy_anisotropy = 1.0 - entropy_isotropy
            
            # Method 5: dominance of the first principal component
            primary_dominance = (
                eigenvalues[0] / np.sum(eigenvalues[1:]) if len(eigenvalues) > 1 else 1
            )
            dominance_anisotropy = np.tanh(np.log(primary_dominance + 1) / 2.0)
            
            # Method 6: inverse effective dimensionality
            participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
            effective_dim_anisotropy = 1.0 - (participation_ratio / len(eigenvalues))
            
            # Weighted combination
            anisotropy_components = [
                enhanced_ellipticity * 0.25,
                enhanced_condition * 0.25,
                ratio_anisotropy * 0.20,
                entropy_anisotropy * 0.15,
                dominance_anisotropy * 0.10,
                effective_dim_anisotropy * 0.05,
            ]
            
            weighted_anisotropy = np.sum(anisotropy_components)
            
            # Adjust based on data type
            if self.isotropy_preference == "low":
                # trajectories favor high anisotropy
                score = weighted_anisotropy
            else:
                # steady state favors low anisotropy (high isotropy)
                score = 1.0 - weighted_anisotropy
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"Error in isotropy/anisotropy analysis: {e}")
            return 0.5


    # ==================== 3. Single-cell–specific metrics ====================
    
    def trajectory_directionality_score(self, latent_space):
        """
        Trajectory directionality score.
        Quantifies how strongly a dominant developmental axis is expressed.
        """
        try:
            pca = PCA()
            pca.fit(latent_space)
            explained_var = pca.explained_variance_ratio_
            
            if len(explained_var) >= 2:
                main_dominance = explained_var[0]
                
                other_variance = np.sum(explained_var[1:])
                if other_variance > 1e-10:
                    dominance_ratio = explained_var[0] / other_variance
                    directionality = dominance_ratio / (1.0 + dominance_ratio)
                else:
                    directionality = 1.0
            else:
                directionality = 1.0
                
            return np.clip(directionality, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"Error computing trajectory directionality: {e}")
            return 0.5
    
    def noise_resilience_score(self, latent_space):
        """
        Noise resilience score.
        Assesses how well the latent space filters out technical noise.
        """
        try:
            pca = PCA()
            pca.fit(latent_space)
            explained_variance = pca.explained_variance_
            
            if len(explained_variance) > 1:
                signal_variance = np.sum(explained_variance[:2])  # first two PCs
                noise_variance = np.sum(explained_variance[2:]) if len(explained_variance) > 2 else 0
                
                if noise_variance > 1e-10:
                    snr = signal_variance / noise_variance
                    noise_resilience = min(snr / 10.0, 1.0)  # normalized
                else:
                    noise_resilience = 1.0
            else:
                noise_resilience = 1.0
                
            return np.clip(noise_resilience, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"Error computing noise resilience: {e}")
            return 0.5
    
    # ==================== 4. Comprehensive evaluation framework ====================
    
    def comprehensive_evaluation(self, latent_space):
        """
        Comprehensive latent space evaluation for single-cell data.
        
        Args:
            latent_space: latent coordinates
            
        Returns:
            dict: full set of evaluation results
        """
        
        self._log(f"Starting comprehensive evaluation for single-cell data ({self.data_type})...")
        
        results = {}
        
        # 1. Core manifold metrics
        self._log("Computing manifold dimensionality metrics...")
        results['manifold_dimensionality'] = self.manifold_dimensionality_score_v2(latent_space)
        
        # 2. Spectral metrics
        self._log("Computing spectral metrics...")
        results['spectral_decay_rate'] = self.spectral_decay_rate(latent_space)
        results['participation_ratio'] = self.participation_ratio_score(latent_space)
        results['anisotropy_score'] = self.isotropy_anisotropy_score(latent_space)
        
        # 3. Single-cell–specific metrics
        self._log("Computing single-cell–specific metrics...")
        results['trajectory_directionality'] = self.trajectory_directionality_score(latent_space)
        
        # 4. Technical quality metrics
        self._log("Computing technical quality metrics...")
        results['noise_resilience'] = self.noise_resilience_score(latent_space)
        
        # 5. Aggregate scores
        self._log("Computing aggregate scores...")
        
        core_metrics = [
            results['manifold_dimensionality'],
            results['spectral_decay_rate'],
            results['participation_ratio'],
            results['anisotropy_score'],
        ]
        results['core_quality'] = np.mean(core_metrics)
        
        if self.data_type == "trajectory":
            # Trajectories: emphasize directionality
            final_components = [
                results['core_quality'] * 0.5,
                results['trajectory_directionality'] * 0.3,
                results['noise_resilience'] * 0.2,
            ]
        else:
            # Steady state: emphasize core manifold quality
            final_components = [
                results['core_quality'] * 0.7,
                results['noise_resilience'] * 0.3,
            ]
        
        results['overall_quality'] = np.sum(final_components)
        
        # Add interpretation
        results['data_type'] = self.data_type
        results['interpretation'] = self._generate_interpretation(results)
        
        if self.verbose:
            self._print_comprehensive_results(results)
        
        return results
    
    def _generate_interpretation(self, results):
        """Generate a qualitative interpretation of the results."""
        
        interpretation = {
            'quality_level': '',
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
        }
        
        overall = results['overall_quality']
        
        # Quality level
        if overall >= 0.8:
            interpretation['quality_level'] = "Excellent"
        elif overall >= 0.6:
            interpretation['quality_level'] = "Good"
        elif overall >= 0.4:
            interpretation['quality_level'] = "Fair"
        else:
            interpretation['quality_level'] = "Needs improvement"
        
        thresholds = {'high': 0.7, 'medium': 0.5, 'low': 0.3}
        
        # Strengths
        if results['manifold_dimensionality'] > thresholds['high']:
            interpretation['strengths'].append("High dimensional compression efficiency")
        
        if results['spectral_decay_rate'] > thresholds['high']:
            interpretation['strengths'].append("Strong eigenvalue decay")
            
        if results['anisotropy_score'] > thresholds['high']:
            if self.data_type == "trajectory":
                interpretation['strengths'].append("Strong trajectory directionality")
            else:
                interpretation['strengths'].append("Uniform spatial distribution")
                
        if results['participation_ratio'] > thresholds['high']:
            if self.data_type == "trajectory":
                interpretation['strengths'].append("High information concentration")
            else:
                interpretation['strengths'].append("Balanced use of dimensions")
        
        if results['trajectory_directionality'] > thresholds['high']:
            interpretation['strengths'].append("Clear dominant developmental axis")
        
        # Weaknesses
        if results['noise_resilience'] < thresholds['medium']:
            interpretation['weaknesses'].append("Insufficient noise filtering")
            
        if results['trajectory_directionality'] < thresholds['medium']:
            interpretation['weaknesses'].append("Dominant developmental axis is weak")
            
        if results['core_quality'] < thresholds['medium']:
            interpretation['weaknesses'].append("Low core manifold quality")
        
        # Recommendations
        if overall < 0.6:
            interpretation['recommendations'].append("Consider adjusting dimensionality reduction parameters")
            interpretation['recommendations'].append("Add or refine preprocessing steps")
            
        if results['noise_resilience'] < 0.4:
            interpretation['recommendations'].append("Strengthen noise filtering")
            
        if self.data_type == "trajectory" and results['trajectory_directionality'] < 0.5:
            interpretation['recommendations'].append("Optimize preservation of trajectory directionality")
        
        return interpretation
    
    def _print_comprehensive_results(self, results):
        """Print human-readable summary of the evaluation."""
        
        print("\n" + "=" * 80)
        print(f"     Single-cell ({self.data_type.upper()}) Latent Space Quality Evaluation")
        print("=" * 80)
        
        # Core metrics
        print(f"\n[Core Manifold Metrics]")
        print(f"  Manifold dimensionality consistency: {results['manifold_dimensionality']:.4f} ★")
        print(f"  Spectral decay rate: {results['spectral_decay_rate']:.4f} (higher is better)")
        print(
            f"  Participation ratio score: {results['participation_ratio']:.4f} "
            f"({'lower is better' if self.participation_preference == 'low' else 'higher is better'})"
        )
        print(
            f"  Anisotropy score: {results['anisotropy_score']:.4f} "
            f"({'higher anisotropy preferred' if self.isotropy_preference == 'low' else 'lower anisotropy preferred'})"
        )
        
        # Single-cell–specific
        print(f"\n[Single-cell Specific Metrics]")
        print(f"  Trajectory directionality: {results['trajectory_directionality']:.4f} (higher is better)")
        
        # Technical quality
        print(f"\n[Technical Quality Metrics]")
        print(f"  Noise resilience: {results['noise_resilience']:.4f} (higher is better)")
        
        # Aggregate
        print(f"\n[Aggregate Scores]")
        print(f"  Core quality score: {results['core_quality']:.4f}")
        print(f"  Overall quality score: {results['overall_quality']:.4f} ★★★")
        
        # Interpretation
        interp = results['interpretation']
        print(f"\n[Interpretation]")
        print(f"  Quality level: {interp['quality_level']}")
        
        if interp['strengths']:
            print(f"  Strengths: {', '.join(interp['strengths'])}")
        
        if interp['weaknesses']:
            print(f"  Weaknesses: {', '.join(interp['weaknesses'])}")
            
        if interp['recommendations']:
            print(f"  Recommendations: {', '.join(interp['recommendations'])}")
        
        print("=" * 80)
    
    def compare_methods(self, method_results_dict):
        """
        Compare different dimensionality reduction methods.
        
        Args:
            method_results_dict: dict mapping method name to latent_space
        
        Returns:
            DataFrame: comparison table of evaluation metrics
        """
        
        comparison_results = []
        
        for method_name, latent_space in method_results_dict.items():
            self._log(f"\nEvaluating method: {method_name}")
            
            # Temporarily disable verbose printing
            original_verbose = self.verbose
            self.verbose = False
            
            results = self.comprehensive_evaluation(latent_space)
            
            # Restore verbosity setting
            self.verbose = original_verbose
            
            comparison_results.append({
                'Method': method_name,
                'Overall_Quality': results['overall_quality'],
                'Manifold_Dimensionality': results['manifold_dimensionality'],
                'Spectral_Decay': results['spectral_decay_rate'],
                'Participation_Ratio': results['participation_ratio'],
                'Anisotropy_Score': results['anisotropy_score'],
                'Trajectory_Directionality': results['trajectory_directionality'],
                'Noise_Resilience': results['noise_resilience'],
                'Quality_Level': results['interpretation']['quality_level'],
            })
        
        df = pd.DataFrame(comparison_results)
        
        # Sort by overall quality
        df = df.sort_values('Overall_Quality', ascending=False)
        
        if self.verbose:
            self._print_comparison_table(df)
        
        return df
    
    def _print_comparison_table(self, df):
        """Print a formatted comparison table."""
        
        print(f"\n{'=' * 100}")
        print(f"                 Dimensionality Reduction Method Comparison ({self.data_type.upper()} data)")
        print('=' * 100)
        
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print(df.to_string(index=False))
        
        print(f"\nBest method: {df.iloc[0]['Method']} (Overall score: {df.iloc[0]['Overall_Quality']:.4f})")
        
        print('=' * 100)


# ==================== Convenience functions ====================

def evaluate_single_cell_latent_space(latent_space, data_type="trajectory", verbose=True):
    """
    Convenience function: evaluate the quality of a single-cell latent space.
    
    Args:
        latent_space: latent coordinates
        data_type: "trajectory" or "steady_state"
        verbose: whether to print detailed logs
        
    Returns:
        dict: evaluation results
    """
    
    evaluator = SingleCellLatentSpaceEvaluator(data_type=data_type, verbose=verbose)
    return evaluator.comprehensive_evaluation(latent_space)


def compare_single_cell_methods(method_results_dict, data_type="trajectory", verbose=True):
    """
    Convenience function: compare multiple single-cell dimensionality reduction methods.
    
    Args:
        method_results_dict: dict mapping method name to latent_space
        data_type: "trajectory" or "steady_state"
        verbose: whether to print detailed logs
        
    Returns:
        DataFrame: comparison results
    """
    
    evaluator = SingleCellLatentSpaceEvaluator(data_type=data_type, verbose=verbose)
    return evaluator.compare_methods(method_results_dict)