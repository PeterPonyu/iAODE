from typing import Callable, Dict, List, Optional, Tuple
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress
from latent_space_simulator import LatentSpaceSimulator


class MetricValidator:
    """
    Validate metrics against ground truth using simulated data
    """
    
    def __init__(self, simulator: LatentSpaceSimulator):
        self.simulator = simulator
    
    def validate_single_metric_clusters(self,
                                        metric_function: Callable,
                                        metric_name: str,
                                        continuity_range: np.ndarray = np.linspace(0.0, 1.0, 11),
                                        n_replicates: int = 6,
                                        n_clusters: int = 5,
                                        n_cells: int = 500,
                                        n_dims: int = 50,
                                        base_separation: float = 5.0,
                                        target_trajectory: str = 'linear',
                                        branch_point: float = 0.5) -> pd.DataFrame:
        """
        Validate a single metric across cluster continuity levels
        
        Parameters:
            metric_function: Function that takes latent_space and returns score
            metric_name: Name of the metric
            continuity_range: Range of continuity values to test (0.0 to 1.0)
            n_replicates: Number of replicates per continuity level
            n_clusters: Number of clusters
            n_cells: Number of cells to simulate
            n_dims: Dimensionality of latent space
            base_separation: Base distance between cluster centers
            target_trajectory: 'linear', 'branching', or 'cyclic'
            
        Returns:
            DataFrame with validation results
        """
        results = []
        
        for continuity in continuity_range:
            for rep in range(n_replicates):
                # Generate discrete cluster data
                sim_data = self.simulator.simulate_discrete_clusters(
                    n_cells=n_cells,
                    n_dims=n_dims,
                    n_clusters=n_clusters,
                    continuity=continuity,
                    base_separation=base_separation,
                    target_trajectory=target_trajectory,
                    branch_point=branch_point,
                    return_metadata=True
                )
                
                latent_space = sim_data['latent_space']
                
                # Calculate metric
                try:
                    metric_score = metric_function(latent_space)
                except Exception as e:
                    warnings.warn(f"Metric calculation failed: {e}")
                    metric_score = np.nan
                
                results.append({
                    'continuity_setting': continuity,
                    'metric_score': metric_score,
                    'metric_name': metric_name,
                    'replicate': rep,
                    'trajectory_type': 'discrete',
                    'n_clusters': n_clusters,
                    'target_trajectory': target_trajectory
                })
        
        return pd.DataFrame(results)
    
    
    def validate_multiple_metrics_clusters(self,
                                           metric_dict: Dict[str, Callable],
                                           continuity_range: np.ndarray = np.linspace(0.0, 1.0, 11),
                                           n_replicates: int = 6,
                                           n_clusters_list: List[int] = [3, 5, 8],
                                           n_cells: int = 500,
                                           n_dims: int = 50,
                                           base_separation: float = 5.0,
                                           target_trajectory: str = 'linear',
                                           branch_point: float = 0.5) -> pd.DataFrame:
        """
        Validate multiple metrics on discrete clusters with varying continuity
        
        Parameters:
            metric_dict: Dictionary of {metric_name: metric_function}
            continuity_range: Range of continuity values to test
            n_replicates: Number of replicates per condition
            n_clusters_list: List of cluster numbers to test
            n_cells: Number of cells to simulate
            n_dims: Dimensionality of latent space
            base_separation: Base distance between cluster centers
            target_trajectory: 'linear', 'branching', or 'cyclic'
        """
        all_results = []
        
        for n_clusters in n_clusters_list:
            print(f"\nValidating on {n_clusters} clusters...")
            for metric_name, metric_func in metric_dict.items():
                print(f"  Testing {metric_name}...")
                df = self.validate_single_metric_clusters(
                    metric_func, metric_name, continuity_range, 
                    n_replicates, n_clusters, n_cells, n_dims,
                    base_separation, target_trajectory, branch_point
                )
                all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True)
    
    
    def validate_single_metric(self,
                              metric_function: Callable,
                              metric_name: str,
                              continuity_range: np.ndarray = np.linspace(0.9, 0.99, 9),
                              n_replicates: int = 6,
                              trajectory_type: str = 'linear',
                              n_cells: int = 500,
                              n_dims: int = 50,
                              noise_type: str = 'gaussian',
                              branch_point: float = 0.4,
                              n_branches: int = 2,
                              branch_angle: float = 60,
                              n_cycles: float = 1.5) -> pd.DataFrame:
        """
        Validate a single metric across continuity levels
        
        Parameters:
            metric_function: Function that takes latent_space and returns score
            metric_name: Name of the metric
            continuity_range: Range of continuity values to test
            n_replicates: Number of replicates per continuity level
            trajectory_type: 'linear', 'branching', or 'cyclic'
            n_cells: Number of cells to simulate
            n_dims: Dimensionality of latent space
            noise_type: Noise type for linear trajectory ('gaussian', 'uniform', 'heavy_tail')
            branch_point: Branch point for branching trajectory
            n_branches: Number of branches for branching trajectory
            branch_angle: Angle between branches
            n_cycles: Number of cycles for cyclic trajectory
            
        Returns:
            DataFrame with validation results
        """
        results = []
        
        for continuity in continuity_range:
            for rep in range(n_replicates):
                # Generate data
                if trajectory_type == 'linear':
                    sim_data = self.simulator.simulate_linear_trajectory(
                        n_cells=n_cells,
                        n_dims=n_dims,
                        continuity=continuity,
                        noise_type=noise_type,
                        return_metadata=True
                    )
                elif trajectory_type == 'branching':
                    sim_data = self.simulator.simulate_branching_trajectory(
                        n_cells=n_cells,
                        n_dims=n_dims,
                        continuity=continuity,
                        branch_point=branch_point,
                        n_branches=n_branches,
                        branch_angle=branch_angle,
                        return_metadata=True
                    )
                elif trajectory_type == 'cyclic':
                    sim_data = self.simulator.simulate_cyclic_trajectory(
                        n_cells=n_cells,
                        n_dims=n_dims,
                        continuity=continuity,
                        n_cycles=n_cycles,
                        return_metadata=True
                    )
                
                latent_space = sim_data['latent_space']
                
                # Calculate metric
                try:
                    metric_score = metric_function(latent_space)
                except Exception as e:
                    warnings.warn(f"Metric calculation failed: {e}")
                    metric_score = np.nan
                
                results.append({
                    'continuity_setting': continuity,
                    'metric_score': metric_score,
                    'metric_name': metric_name,
                    'replicate': rep,
                    'trajectory_type': trajectory_type
                })
        
        return pd.DataFrame(results)
    
    
    def validate_multiple_metrics(self,
                                  metric_dict: Dict[str, Callable],
                                  continuity_range: np.ndarray = np.linspace(0.9, 0.99, 9),
                                  n_replicates: int = 6,
                                  trajectory_types: List[str] = ['linear', 'branching', 'cyclic'],
                                  n_cells: int = 500,
                                  n_dims: int = 50,
                                  noise_type: str = 'gaussian',
                                  branch_point: float = 0.4,
                                  n_branches: int = 2,
                                  branch_angle: float = 60,
                                  n_cycles: float = 1.5) -> pd.DataFrame:
        """
        Validate multiple metrics simultaneously
        
        Parameters:
            metric_dict: Dictionary of {metric_name: metric_function}
            continuity_range: Range of continuity values to test
            n_replicates: Number of replicates per condition
            trajectory_types: List of trajectory types to test
            n_cells: Number of cells to simulate
            n_dims: Dimensionality of latent space
            noise_type: Noise type for linear trajectory
            branch_point: Branch point for branching trajectory
            n_branches: Number of branches for branching trajectory
            branch_angle: Angle between branches
            n_cycles: Number of cycles for cyclic trajectory
        """
        all_results = []
        
        for traj_type in trajectory_types:
            print(f"\nValidating on {traj_type} trajectory...")
            for metric_name, metric_func in metric_dict.items():
                print(f"  Testing {metric_name}...")
                df = self.validate_single_metric(
                    metric_func, metric_name, continuity_range, 
                    n_replicates, traj_type, n_cells, n_dims,
                    noise_type, branch_point, n_branches, branch_angle, n_cycles
                )
                all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True)
        
    def calculate_metric_sensitivity(self, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sensitivity of metrics to continuity changes
        
        Metrics:
        - Correlation with true continuity
        - Dynamic range (max - min)
        - Monotonicity
        """
        sensitivity_results = []
        
        for metric_name in validation_df['metric_name'].unique():
            for traj_type in validation_df['trajectory_type'].unique():
                subset = validation_df[
                    (validation_df['metric_name'] == metric_name) &
                    (validation_df['trajectory_type'] == traj_type)
                ]
                
                # Dynamic range
                dynamic_range = subset['metric_score'].max() - subset['metric_score'].min()
                
                sensitivity_results.append({
                    'metric_name': metric_name,
                    'trajectory_type': traj_type,
                    'dynamic_range': dynamic_range,
                    'mean_score': subset['metric_score'].mean(),
                    'std_score': subset['metric_score'].std()
                })
        
        return pd.DataFrame(sensitivity_results)
    
    def analyze_trend_significance(self, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform statistical analysis on metric trends with increasing continuity
        Works for both trajectory and cluster validation data
        
        Tests performed:
        - Pearson correlation: Linear relationship strength
        - Spearman correlation: Monotonic relationship (rank-based)
        - Linear regression: Slope and significance
        - Effect size: Normalized change per unit continuity
        
        Parameters:
            validation_df: DataFrame from validate_single_metric or validate_multiple_metrics
            
        Returns:
            DataFrame with statistical test results for each metric-trajectory/cluster combination
        """
        from scipy.stats import pearsonr, spearmanr, linregress
        
        # Detect data type and set grouping variable
        if 'n_clusters' in validation_df.columns:
            row_var = 'n_clusters'
            row_values = sorted(validation_df[row_var].unique())
        else:
            row_var = 'trajectory_type'
            row_values = validation_df[row_var].unique()
        
        trend_results = []
        
        for metric_name in validation_df['metric_name'].unique():
            for row_value in row_values:
                subset = validation_df[
                    (validation_df['metric_name'] == metric_name) &
                    (validation_df[row_var] == row_value)
                ].copy()
                
                # Remove NaN values
                subset = subset.dropna(subset=['metric_score', 'continuity_setting'])
                
                if len(subset) < 3:
                    continue
                
                continuity = subset['continuity_setting'].values
                scores = subset['metric_score'].values
                
                # Pearson correlation (linear relationship)
                pearson_r, pearson_p = pearsonr(continuity, scores)
                
                # Spearman correlation (monotonic relationship)
                spearman_r, spearman_p = spearmanr(continuity, scores)
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = linregress(continuity, scores)
                
                # Effect size: normalized slope
                score_range = scores.max() - scores.min()
                continuity_range = continuity.max() - continuity.min()
                normalized_slope = slope * continuity_range / (score_range + 1e-10)
                
                # Aggregate by continuity level for trend direction analysis
                mean_by_continuity = subset.groupby('continuity_setting')['metric_score'].mean()
                continuity_levels = mean_by_continuity.index.values
                mean_scores = mean_by_continuity.values
                
                # Monotonicity: fraction of increasing consecutive pairs
                if len(mean_scores) > 1:
                    diffs = np.diff(mean_scores)
                    monotonicity = np.sum(diffs > 0) / len(diffs)
                else:
                    monotonicity = np.nan
                
                result = {
                    'metric_name': metric_name,
                    row_var: row_value,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'regression_slope': slope,
                    'regression_p': p_value,
                    'regression_r2': r_value**2,
                    'effect_size': normalized_slope,
                    'monotonicity': monotonicity,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                    'significant_linear': 'Yes' if pearson_p < 0.05 else 'No',
                    'significant_monotonic': 'Yes' if spearman_p < 0.05 else 'No'
                }
                
                trend_results.append(result)
        
        result_df = pd.DataFrame(trend_results)
        
        # Add interpretive summary column
        if len(result_df) > 0:
            result_df['trend_strength'] = pd.cut(
                result_df['spearman_r'].abs(),
                bins=[0, 0.3, 0.7, 0.9, 1.0],
                labels=['weak', 'moderate', 'strong', 'very_strong']
            )
        
        return result_df
    
    def plot_validation_results(self, validation_df: pd.DataFrame, 
                               trend_stats: Optional[pd.DataFrame] = None,
                               save_path: Optional[str] = None,
                               figsize: Optional[Tuple[int, int]] = None,
                               title_fontsize: int = 12,
                               title_fontweight: str = 'normal',
                               label_fontsize: int = 9,
                               label_fontweight: str = 'normal',
                               tick_fontsize: int = 9,
                               legend_fontsize: int = 9,
                               stats_fontsize: int = 9,
                               legend_loc: str = 'lower right',
                               hspace: float = 0.3,
                               wspace: float = 0.3,
                               ylabel_indices: Optional[List[int]] = None,
                               xlabel_indices: Optional[List[int]] = None,
                               title_indices: Optional[List[int]] = None,
                               show_xticklabels: bool = True,
                               show_yticklabels: bool = True):
        """
        Visualize validation results with statistical annotations
        Works for both trajectory and cluster validation data
        
        Parameters:
            validation_df: DataFrame from validation
            trend_stats: DataFrame from analyze_trend_significance (optional)
            save_path: Path to save figure
            figsize: Figure size as (width, height). Default: (5*n_metrics, 4.5*n_rows)
            title_fontsize: Font size for subplot titles
            title_fontweight: Font weight for subplot titles ('normal', 'bold', etc.)
            label_fontsize: Font size for axis labels
            label_fontweight: Font weight for axis labels
            tick_fontsize: Font size for axis ticks
            legend_fontsize: Font size for legend text
            stats_fontsize: Font size for statistics text
            legend_loc: Legend location ('lower right', 'upper left', etc.)
            hspace: Height space between subplots
            wspace: Width space between subplots
            ylabel_indices: List of subplot indices to show y-labels (None = all, [] = none)
            xlabel_indices: List of subplot indices to show x-labels (None = all, [] = none)
            title_indices: List of subplot indices to show titles (None = all, [] = none)
            show_xticklabels: Whether to show x-axis tick labels
            show_yticklabels: Whether to show y-axis tick labels
        """
        from scipy import stats
        
        # Compute trend stats if not provided
        if trend_stats is None:
            trend_stats = self.analyze_trend_significance(validation_df)
        
        # Detect data type and set grouping variable
        if 'n_clusters' in validation_df.columns:
            # Cluster validation data
            row_var = 'n_clusters'
            row_values = sorted(validation_df[row_var].unique())
            row_label_format = lambda x: f'{x} Clusters'
            xlabel_text = 'Continuity Setting'
        else:
            # Trajectory validation data
            row_var = 'trajectory_type'
            row_values = validation_df[row_var].unique()
            row_label_format = lambda x: x.capitalize()
            xlabel_text = 'Continuity Setting'
        
        metrics = validation_df['metric_name'].unique()
        
        n_metrics = len(metrics)
        n_rows = len(row_values)
        
        # Set default figure size if not provided
        if figsize is None:
            figsize = (5*n_metrics, 4.5*n_rows)
        
        fig, axes = plt.subplots(n_rows, n_metrics, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_metrics == 1:
            axes = axes.reshape(-1, 1)
        
        # Adjust subplot spacing
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        
        # Default behavior: show labels on appropriate edges
        if ylabel_indices is None:
            # Show ylabel on first column only
            ylabel_indices = list(range(0, n_rows * n_metrics, n_metrics))
        if xlabel_indices is None:
            # Show xlabel on bottom row only
            xlabel_indices = list(range((n_rows - 1) * n_metrics, n_rows * n_metrics))
        if title_indices is None:
            # Show all titles by default
            title_indices = list(range(n_metrics))
        
        for i, row_value in enumerate(row_values):
            for j, metric_name in enumerate(metrics):
                ax = axes[i, j]
                
                # Calculate flat index for this subplot
                flat_idx = i * n_metrics + j
                
                subset = validation_df[
                    (validation_df['metric_name'] == metric_name) &
                    (validation_df[row_var] == row_value)
                ].copy()
                
                # Get statistical results for this combination
                stat_row = trend_stats[
                    (trend_stats['metric_name'] == metric_name) &
                    (trend_stats[row_var] == row_value)
                ]
                
                # Calculate mean and SEM per continuity level
                grouped = subset.groupby('continuity_setting')['metric_score']
                mean_scores = grouped.mean()
                sem_scores = grouped.sem()
                continuity_levels = mean_scores.index.values
                
                # Plot mean with error shading
                ax.plot(continuity_levels, mean_scores.values, 
                       'o-', linewidth=2.5, color='#1f77b4', 
                       label='Mean', markersize=6, zorder=2)
                
                # Add error bars
                if not sem_scores.isna().all():
                    ax.fill_between(continuity_levels,
                                   mean_scores.values - sem_scores.values,
                                   mean_scores.values + sem_scores.values,
                                   alpha=0.2, color='#1f77b4', zorder=1)
                
                # Add regression line if significant
                if len(stat_row) > 0:
                    row = stat_row.iloc[0]
                    
                    # Plot regression line if trend is significant
                    if row['regression_p'] < 0.05:
                        x_fit = np.linspace(continuity_levels.min(), continuity_levels.max(), 100)
                        y_fit = row['regression_slope'] * x_fit + (mean_scores.mean() - row['regression_slope'] * continuity_levels.mean())
                        ax.plot(x_fit, y_fit, '--', color='red', linewidth=1.5, 
                               alpha=0.7, label='Linear fit', zorder=3)
                    
                    # Create statistical annotation
                    sig_marker = ''
                    if row['spearman_p'] < 0.001:
                        sig_marker = '***'
                    elif row['spearman_p'] < 0.01:
                        sig_marker = '**'
                    elif row['spearman_p'] < 0.05:
                        sig_marker = '*'
                    else:
                        sig_marker = 'n.s.'
                    
                    # Format statistics text
                    stats_text = (
                        f"ρ = {row['spearman_r']:.3f} {sig_marker}\n"
                        f"R² = {row['regression_r2']:.3f}\n"
                        f"Slope = {row['regression_slope']:.3f}"
                    )
                    
                    # Position text box
                    ax.text(0.05, 0.95, stats_text,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           fontsize=stats_fontsize)
                
                # Conditionally set labels based on indices
                if flat_idx in xlabel_indices:
                    ax.set_xlabel(xlabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax.set_xlabel('')
                
                if flat_idx in ylabel_indices:
                    ylabel_text = f'Metric Score\n({row_label_format(row_value)})'
                    ax.set_ylabel(ylabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax.set_ylabel('')
                
                # Conditionally set title
                if flat_idx in title_indices:
                    ax.set_title(f'{metric_name}', fontsize=title_fontsize, fontweight=title_fontweight)
                else:
                    ax.set_title('')
                
                # Handle tick labels
                if not show_xticklabels:
                    ax.set_xticklabels([])
                else:
                    ax.tick_params(axis='x', labelsize=tick_fontsize)
                
                if not show_yticklabels:
                    ax.set_yticklabels([])
                else:
                    ax.tick_params(axis='y', labelsize=tick_fontsize)
                
                ax.legend(loc=legend_loc, fontsize=legend_fontsize)
                ax.grid(True, alpha=0.3, linestyle='--')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def visualize_two_group_comparison(self,
                                       validation_df: pd.DataFrame,
                                       plot_type: str = 'box',  # 'box' or 'bar'
                                       test_type: str = 'mann-whitney',  # 'mann-whitney' or 't-test'
                                       figsize: Optional[Tuple[int, int]] = None,
                                       save_path: Optional[str] = None,
                                       show_points: bool = True,
                                       title_fontsize: int = 11,
                                       label_fontsize: int = 9,
                                       tick_fontsize: int = 9,
                                       wspace: float = 0.3,
                                       hspace: float = 0.3,
                                       short_title: Optional[dict] = None,  # dict mapping full name to short name
                                       show_spines: bool = False,
                                       hatch_patterns: Optional[list] = None,  # list of hatch patterns
                                       hatch_color: str = 'w',
                                       bar_colors: Optional[list] = None,  # list of bar colors
                                       sig_fontsize: int = 10,
                                       sig_y_offset: float = 0.05,  # offset as fraction of y_range
                                       sig_bar_height: float = 0.015,  # bar height in absolute units
                                       sig_text_offset: float = 0.005):  # text offset above bar in absolute units
        """
        Visualize comparison between two continuity levels with statistical tests
        
        Parameters:
            validation_df: DataFrame from validate_single/multiple_metrics_clusters
            plot_type: 'box' or 'bar'
            test_type: 'mann-whitney' or 't-test'
            figsize: Figure size
            save_path: Path to save figure
            show_points: Whether to show individual data points
            wspace: Width space between subplots
            hspace: Height space between subplots
            short_title: Dict mapping metric names to short names for titles
            show_spines: Whether to show top and right spines
            hatch_patterns: List of hatch patterns for bars (e.g., ['///', '\\\\'])
            hatch_color: Color for hatch patterns
            bar_colors: List of colors for bars/boxes
            sig_fontsize: Font size for significance markers
            sig_y_offset: Vertical offset for significance bar (fraction of y_range)
            sig_bar_height: Height of significance bar in absolute data units
            sig_text_offset: Vertical offset for text above bar in absolute data units
            
        Returns:
            fig, axes, stats_results
        """
        from scipy import stats
        
        # Check that we have exactly 2 continuity levels
        continuity_levels = validation_df['continuity_setting'].unique()
        if len(continuity_levels) != 2:
            raise ValueError(f"Expected 2 continuity levels, got {len(continuity_levels)}")
        
        # Get unique metrics and cluster settings
        metrics = validation_df['metric_name'].unique()
        
        if 'n_clusters' in validation_df.columns:
            n_clusters_list = sorted(validation_df['n_clusters'].unique())
        else:
            n_clusters_list = [None]
    
        default_short_titles = {
            'Spectral Decay': 'Spec. Decay',
            'Anisotropy': 'Aniso.',
            'Participation Ratio': 'Part. Ratio',
            'Trajectory Directionality': 'Traj. Direct.',
            'Manifold Dimensionality': 'Manif. Dim.',
            'Noise resilience': 'Noise Resil.'
        }
        
        # Use provided short_title or default
        if short_title is None:
            short_title = default_short_titles
        else:
            # Merge with defaults
            short_title = {**default_short_titles, **short_title}
        
        # Default hatch patterns
        if hatch_patterns is None:
            hatch_patterns = ['/', '/']
        
        # Default bar colors
        if bar_colors is None:
            bar_colors = ['lightblue', 'lightcoral']
        
        # Determine subplot layout
        n_rows = len(n_clusters_list)
        n_cols = len(metrics)
        
        # Set figure size
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        
        stats_results = []
        
        # Plot for each condition
        for row_idx, n_clust in enumerate(n_clusters_list):
            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                
                # Filter data
                mask = validation_df['metric_name'] == metric
                if n_clust is not None:
                    mask &= validation_df['n_clusters'] == n_clust
                
                data = validation_df[mask]
                
                # Prepare data for plotting
                group_0 = data[data['continuity_setting'] == continuity_levels[0]]['metric_score'].values
                group_1 = data[data['continuity_setting'] == continuity_levels[1]]['metric_score'].values
                
                # Remove NaN values
                group_0 = group_0[~np.isnan(group_0)]
                group_1 = group_1[~np.isnan(group_1)]
                
                # Statistical test
                if test_type == 'mann-whitney':
                    stat, pval = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                else:  # t-test
                    stat, pval = stats.ttest_ind(group_0, group_1)
                    test_name = 't-test'
                
                # Store results
                stats_results.append({
                    'metric': metric,
                    'n_clusters': n_clust,
                    'test': test_name,
                    'statistic': stat,
                    'p_value': pval,
                    'group_0_mean': np.mean(group_0),
                    'group_1_mean': np.mean(group_1),
                    'group_0_std': np.std(group_0),
                    'group_1_std': np.std(group_1)
                })
                
                # Plotting
                if plot_type == 'box':
                    bp = ax.boxplot([group_0, group_1], 
                                   labels=[f'{continuity_levels[0]:.1f}', f'{continuity_levels[1]:.1f}'],
                                   patch_artist=True,
                                   widths=0.6)
                    
                    # Color boxes
                    for patch, color in zip(bp['boxes'], bar_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Show individual points
                    if show_points:
                        x_pos = [1, 2]
                        for i, group in enumerate([group_0, group_1]):
                            x = np.random.normal(x_pos[i], 0.04, size=len(group))
                            ax.scatter(x, group, alpha=0.4, s=20, c='black', zorder=3)
                
                elif plot_type == 'bar':
                    means = [np.mean(group_0), np.mean(group_1)]
                    stds = [np.std(group_0), np.std(group_1)]
                    x_pos = [0, 1]
                    
                    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                                 color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                    
                    # Add hatch patterns
                    for bar, hatch in zip(bars, hatch_patterns):
                        bar.set_hatch(hatch)
                        bar.set_edgecolor(hatch_color)
                    
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f'{continuity_levels[0]:.1f}', f'{continuity_levels[1]:.1f}'])
                    
                    # Show individual points
                    if show_points:
                        for i, group in enumerate([group_0, group_1]):
                            x = np.random.normal(x_pos[i], 0.04, size=len(group))
                            ax.scatter(x, group, alpha=0.4, s=20, c='black', zorder=3)
                
                # Add significance annotation
                y_max = max(np.max(group_0), np.max(group_1))
                y_min = min(np.min(group_0), np.min(group_1))
                y_range = y_max - y_min
                
                # Significance stars
                if pval < 0.001:
                    sig_text = '***'
                elif pval < 0.01:
                    sig_text = '**'
                elif pval < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'n.s.'
                
                # Draw significance bar
                if plot_type == 'box':
                    x1, x2 = 1, 2
                else:
                    x1, x2 = 0, 1
                
                y_sig = y_max + y_range * sig_y_offset
                y_bar_top = y_sig + sig_bar_height
                ax.plot([x1, x1, x2, x2], 
                       [y_sig, y_bar_top, y_bar_top, y_sig], 
                       'k-', linewidth=1)
                ax.text((x1 + x2) / 2, y_bar_top + sig_text_offset, sig_text, 
                       ha='center', va='bottom', fontsize=sig_fontsize)
                
                # Title: only show metric name at top row
                if row_idx == 0:
                    title_text = short_title.get(metric, metric)
                    ax.set_title(title_text, fontsize=title_fontsize)
                
                # Y-label: only show at left column
                if col_idx == 0:
                    if n_clust is not None:
                        ax.set_ylabel(f'{n_clust} clusters', fontsize=label_fontsize)
                    else:
                        ax.set_ylabel('Metric Score', fontsize=label_fontsize)
                
                # X-label: only show at bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Continuity', fontsize=label_fontsize)
                
                # Handle spine visibility
                if not show_spines:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                
                ax.tick_params(labelsize=tick_fontsize)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig, axes, pd.DataFrame(stats_results)

