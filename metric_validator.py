from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress
from scipy import stats


class MetricValidator:
    """
    Validate metrics against ground truth using simulated data with flexible configuration
    """
    
    def __init__(self, simulator):
        self.simulator = simulator
    
    def validate_single_metric(self,
                              metric_function: Callable,
                              metric_name: str,
                              continuity_range: np.ndarray = np.linspace(0.9, 0.99, 9),
                              n_replicates: int = 6,
                              trajectory_configs: Optional[List[Dict]] = None,
                              n_cells: int = 500,
                              n_dims: int = 50) -> pd.DataFrame:
        """
        Validate a single metric across continuity levels with flexible trajectory configuration.
        
        Parameters:
            metric_function: Function that takes latent_space and returns score
            metric_name: Name of the metric
            continuity_range: Range of continuity values to test
            n_replicates: Number of replicates per continuity level
            trajectory_configs: List of trajectory configurations, each containing:
                - 'type': str - 'linear', 'branching', or 'cyclic'
                - 'params': dict - trajectory-specific parameters
                - 'label': str (optional) - identifier for this config
                
                Example:
                    [
                        {'type': 'linear', 'params': {'noise_type': 'gaussian'}},
                        {'type': 'branching', 'params': {'n_branches': 2, 'branch_point': 0.5}},
                        {'type': 'branching', 'params': {'n_branches': 3, 'branch_point': 0.5}},
                    ]
            
            n_cells: Default number of cells (overridden by params['n_cells'])
            n_dims: Default dimensionality (overridden by params['n_dims'])
            
        Returns:
            DataFrame with validation results including config identifier
        """
        # Default configs if none provided
        if trajectory_configs is None:
            trajectory_configs = [
                {'type': 'linear', 'params': {}},
                {'type': 'branching', 'params': {}},
                {'type': 'cyclic', 'params': {}},
            ]
        
        results = []
        
        for config_idx, config in enumerate(trajectory_configs):
            traj_type = config['type']
            traj_params = {'n_cells': n_cells, 'n_dims': n_dims, **config.get('params', {})}
            
            # Generate config label
            if 'label' in config:
                config_label = config['label']
            else:
                # Auto-generate label from params
                param_strs = [f"{k}={v}" for k, v in config.get('params', {}).items() 
                             if k not in ['n_cells', 'n_dims']]
                if param_strs:
                    config_label = f"{traj_type}_{','.join(param_strs)}"
                else:
                    config_label = traj_type
            
            print(f"  Config: {config_label}...")
            
            for continuity in continuity_range:
                for rep in range(n_replicates):
                    # Generate data
                    sim_data = self._generate_trajectory(
                        traj_type=traj_type,
                        continuity=continuity,
                        params=traj_params
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
                        'trajectory_type': traj_type,
                        'config_label': config_label,
                        'config_idx': config_idx,
                        **{f'param_{k}': v for k, v in config.get('params', {}).items()}
                    })
        
        return pd.DataFrame(results)
    
    def validate_multiple_metrics(self,
                                  metric_dict: Dict[str, Callable],
                                  continuity_range: np.ndarray = np.linspace(0.9, 0.99, 9),
                                  n_replicates: int = 6,
                                  trajectory_configs: Optional[List[Dict]] = None,
                                  n_cells: int = 500,
                                  n_dims: int = 50) -> pd.DataFrame:
        """
        Validate multiple metrics with flexible trajectory configuration.
        
        Parameters:
            metric_dict: Dictionary of {metric_name: metric_function}
            continuity_range: Range of continuity values to test
            n_replicates: Number of replicates per condition
            trajectory_configs: List of trajectory configurations (see validate_single_metric)
            n_cells: Default number of cells
            n_dims: Default dimensionality
        """
        all_results = []
        
        for metric_name, metric_func in metric_dict.items():
            print(f"\nTesting {metric_name}...")
            df = self.validate_single_metric(
                metric_func, metric_name, continuity_range, 
                n_replicates, trajectory_configs, n_cells, n_dims
            )
            all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True)
    
    def validate_single_metric_clusters(self,
                                       metric_function: Callable,
                                       metric_name: str,
                                       continuity_range: np.ndarray = np.linspace(0.0, 1.0, 11),
                                       n_replicates: int = 6,
                                       cluster_configs: Optional[List[Dict]] = None,
                                       n_cells: int = 500,
                                       n_dims: int = 50) -> pd.DataFrame:
        """
        Validate a single metric on clusters with flexible configuration.
        
        Parameters:
            metric_function: Function that takes latent_space and returns score
            metric_name: Name of the metric
            continuity_range: Range of continuity values (0.0 to 1.0)
            n_replicates: Number of replicates per continuity level
            cluster_configs: List of cluster configurations, each containing:
                - 'n_clusters': int - number of clusters
                - 'target_trajectory': str - 'linear', 'branching', or 'cyclic'
                - 'params': dict (optional) - additional parameters
                - 'label': str (optional) - identifier for this config
                
                Example:
                    [
                        {'n_clusters': 3, 'target_trajectory': 'linear'},
                        {'n_clusters': 5, 'target_trajectory': 'branching', 
                         'params': {'branch_point': 0.3}},
                        {'n_clusters': 5, 'target_trajectory': 'branching', 
                         'params': {'branch_point': 0.7}},
                    ]
            
            n_cells: Default number of cells
            n_dims: Default dimensionality
            
        Returns:
            DataFrame with validation results
        """
        # Default configs if none provided
        if cluster_configs is None:
            cluster_configs = [
                {'n_clusters': 3, 'target_trajectory': 'linear'},
                {'n_clusters': 5, 'target_trajectory': 'linear'},
                {'n_clusters': 8, 'target_trajectory': 'linear'},
            ]
        
        results = []
        
        for config_idx, config in enumerate(cluster_configs):
            n_clusters = config['n_clusters']
            target_traj = config['target_trajectory']
            
            # Default parameters
            default_params = {
                'n_cells': n_cells,
                'n_dims': n_dims,
                'base_separation': 5.0,
                'branch_point': 0.5,
            }
            merged_params = {**default_params, **config.get('params', {})}
            
            # Generate config label
            if 'label' in config:
                config_label = config['label']
            else:
                param_strs = [f"{k}={v}" for k, v in config.get('params', {}).items()]
                if param_strs:
                    config_label = f"{n_clusters}clust_{target_traj}_{','.join(param_strs)}"
                else:
                    config_label = f"{n_clusters}clust_{target_traj}"
            
            print(f"  Config: {config_label}...")
            
            for continuity in continuity_range:
                for rep in range(n_replicates):
                    # Generate discrete cluster data
                    sim_data = self.simulator.simulate_discrete_clusters(
                        n_clusters=n_clusters,
                        continuity=continuity,
                        target_trajectory=target_traj,
                        return_metadata=True,
                        **merged_params
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
                        'target_trajectory': target_traj,
                        'config_label': config_label,
                        'config_idx': config_idx,
                        **{f'param_{k}': v for k, v in config.get('params', {}).items()}
                    })
        
        return pd.DataFrame(results)
    
    def validate_multiple_metrics_clusters(self,
                                          metric_dict: Dict[str, Callable],
                                          continuity_range: np.ndarray = np.linspace(0.0, 1.0, 11),
                                          n_replicates: int = 6,
                                          cluster_configs: Optional[List[Dict]] = None,
                                          n_cells: int = 500,
                                          n_dims: int = 50) -> pd.DataFrame:
        """
        Validate multiple metrics on clusters with flexible configuration.
        
        Parameters:
            metric_dict: Dictionary of {metric_name: metric_function}
            continuity_range: Range of continuity values to test
            n_replicates: Number of replicates per condition
            cluster_configs: List of cluster configurations (see validate_single_metric_clusters)
            n_cells: Default number of cells
            n_dims: Default dimensionality
        """
        all_results = []
        
        for metric_name, metric_func in metric_dict.items():
            print(f"\nTesting {metric_name}...")
            df = self.validate_single_metric_clusters(
                metric_func, metric_name, continuity_range, 
                n_replicates, cluster_configs, n_cells, n_dims
            )
            all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True)
    
    def calculate_metric_sensitivity(self, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sensitivity of metrics to continuity changes.
        
        Metrics:
        - Correlation with true continuity
        - Dynamic range (max - min)
        - Monotonicity
        """
        sensitivity_results = []
        
        # Determine grouping columns
        group_cols = ['metric_name']
        if 'config_label' in validation_df.columns:
            group_cols.append('config_label')
        elif 'trajectory_type' in validation_df.columns:
            group_cols.append('trajectory_type')
        
        # Group by all relevant columns
        for group_vals, subset in validation_df.groupby(group_cols):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            
            # Dynamic range
            dynamic_range = subset['metric_score'].max() - subset['metric_score'].min()
            
            result = {
                'dynamic_range': dynamic_range,
                'mean_score': subset['metric_score'].mean(),
                'std_score': subset['metric_score'].std()
            }
            
            # Add group identifiers
            for col, val in zip(group_cols, group_vals):
                result[col] = val
            
            sensitivity_results.append(result)
        
        return pd.DataFrame(sensitivity_results)
    
    def analyze_trend_significance(self, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform statistical analysis on metric trends with increasing continuity.
        Works with flexible trajectory/cluster configurations.
        
        Tests performed:
        - Pearson correlation: Linear relationship strength
        - Spearman correlation: Monotonic relationship (rank-based)
        - Linear regression: Slope and significance
        - Effect size: Normalized change per unit continuity
        
        Returns:
            DataFrame with statistical test results for each metric-config combination
        """
        trend_results = []
        
        # Determine grouping columns based on available data
        group_cols = ['metric_name']
        if 'config_label' in validation_df.columns:
            group_cols.append('config_label')
            display_col = 'config_label'
        elif 'n_clusters' in validation_df.columns:
            group_cols.append('n_clusters')
            display_col = 'n_clusters'
        else:
            group_cols.append('trajectory_type')
            display_col = 'trajectory_type'
        
        for group_vals, subset in validation_df.groupby(group_cols):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            
            metric_name = group_vals[0]
            config_value = group_vals[1] if len(group_vals) > 1 else None
            
            # Remove NaN values
            subset = subset.dropna(subset=['metric_score', 'continuity_setting'])
            
            if len(subset) < 3:
                continue
            
            continuity = subset['continuity_setting'].values
            scores = subset['metric_score'].values
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(continuity, scores)
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(continuity, scores)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(continuity, scores)
            
            # Effect size: normalized slope
            score_range = scores.max() - scores.min()
            continuity_range = continuity.max() - continuity.min()
            normalized_slope = slope * continuity_range / (score_range + 1e-10)
            
            # Monotonicity
            mean_by_continuity = subset.groupby('continuity_setting')['metric_score'].mean()
            mean_scores = mean_by_continuity.values
            
            if len(mean_scores) > 1:
                diffs = np.diff(mean_scores)
                monotonicity = np.sum(diffs > 0) / len(diffs)
            else:
                monotonicity = np.nan
            
            result = {
                'metric_name': metric_name,
                display_col: config_value,
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
        
        # Add interpretive summary
        if len(result_df) > 0:
            result_df['trend_strength'] = pd.cut(
                result_df['spearman_r'].abs(),
                bins=[0, 0.3, 0.7, 0.9, 1.0],
                labels=['weak', 'moderate', 'strong', 'very_strong']
            )
        
        return result_df
    
    def plot_validation_results(self, 
                            validation_df: pd.DataFrame, 
                            trend_stats: Optional[pd.DataFrame] = None,
                            save_path: Optional[str] = None,
                            figsize: Optional[Tuple[int, int]] = None,
                            color_scheme: Union[str, Dict[str, str]] = 'default',
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
        Visualize validation results with statistical annotations.
        Works with flexible trajectory/cluster configurations.
        
        Parameters:
            validation_df: DataFrame from validation
            trend_stats: DataFrame from analyze_trend_significance (optional)
            save_path: Path to save figure
            figsize: Figure size (width, height)
            color_scheme: Color scheme for plots. Options:
                - 'default' or 'blue-red': Standard blue (#1f77b4) and red
                - 'light': Light blue and light coral
                - 'teal-orange': Teal (#20B2AA) and orange (#FF8C00)
                - 'purple-green': Purple (#9467BD) and green (#2CA02C)
                - 'navy-crimson': Navy (#000080) and crimson (#DC143C)
                - 'slate-salmon': Slate blue (#6A5ACD) and salmon (#FA8072)
                - Custom dict: {'main': 'color', 'fit': 'color', 'alpha': float}
                    where 'main' is the mean line color, 'fit' is regression line color,
                    and 'alpha' (optional) is the shading transparency (default: 0.2)
            [Other parameters as before]
            
        Example:
            # Use predefined scheme
            validator.plot_validation_results(df, color_scheme='light')
            
            # Use custom colors
            validator.plot_validation_results(
                df, 
                color_scheme={'main': '#FF6B6B', 'fit': '#4ECDC4', 'alpha': 0.3}
            )
        """
        # Define color scheme presets
        color_schemes = {
            'default': {'main': '#1f77b4', 'fit': 'red', 'alpha': 0.2},
            'blue-red': {'main': '#1f77b4', 'fit': 'red', 'alpha': 0.2},
            'light': {'main': 'lightblue', 'fit': 'lightcoral', 'alpha': 0.25},
            'teal-orange': {'main': '#20B2AA', 'fit': '#FF8C00', 'alpha': 0.2},
            'purple-green': {'main': '#9467BD', 'fit': '#2CA02C', 'alpha': 0.2},
            'navy-crimson': {'main': '#000080', 'fit': '#DC143C', 'alpha': 0.2},
            'slate-salmon': {'main': '#6A5ACD', 'fit': '#FA8072', 'alpha': 0.2},
            'cyan-magenta': {'main': '#17BECF', 'fit': '#E377C2', 'alpha': 0.2},
            'olive-pink': {'main': '#BCBD22', 'fit': '#FF69B4', 'alpha': 0.2},
            'steel-coral': {'main': '#4682B4', 'fit': '#FF7F50', 'alpha': 0.2},
        }
        
        # Get color configuration
        if isinstance(color_scheme, str):
            if color_scheme not in color_schemes:
                raise ValueError(
                    f"Unknown color scheme: '{color_scheme}'. "
                    f"Available: {list(color_schemes.keys())}"
                )
            colors = color_schemes[color_scheme]
        elif isinstance(color_scheme, dict):
            # Custom color scheme
            if 'main' not in color_scheme or 'fit' not in color_scheme:
                raise ValueError("Custom color_scheme dict must contain 'main' and 'fit' keys")
            colors = {
                'main': color_scheme['main'],
                'fit': color_scheme['fit'],
                'alpha': color_scheme.get('alpha', 0.2)
            }
        else:
            raise TypeError("color_scheme must be str or dict")
        
        # Compute trend stats if not provided
        if trend_stats is None:
            trend_stats = self.analyze_trend_significance(validation_df)
        
        # Determine grouping strategy based on available columns
        if 'config_label' in validation_df.columns:
            row_var = 'config_label'
            row_values = validation_df[row_var].unique()
            row_label_format = lambda x: x
        elif 'n_clusters' in validation_df.columns:
            row_var = 'n_clusters'
            row_values = sorted(validation_df[row_var].unique())
            row_label_format = lambda x: f'{x} Clusters'
        else:
            row_var = 'trajectory_type'
            row_values = validation_df[row_var].unique()
            row_label_format = lambda x: x.capitalize()
        
        xlabel_text = 'Continuity Setting'
        
        metrics = validation_df['metric_name'].unique()
        
        n_metrics = len(metrics)
        n_rows = len(row_values)
        
        # Set default figure size
        if figsize is None:
            figsize = (5*n_metrics, 4.5*n_rows)
        
        fig, axes = plt.subplots(n_rows, n_metrics, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_metrics == 1:
            axes = axes.reshape(-1, 1)
        
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        
        # Default label behavior
        if ylabel_indices is None:
            ylabel_indices = list(range(0, n_rows * n_metrics, n_metrics))
        if xlabel_indices is None:
            xlabel_indices = list(range((n_rows - 1) * n_metrics, n_rows * n_metrics))
        if title_indices is None:
            title_indices = list(range(n_metrics))
        
        for i, row_value in enumerate(row_values):
            for j, metric_name in enumerate(metrics):
                ax = axes[i, j]
                flat_idx = i * n_metrics + j
                
                subset = validation_df[
                    (validation_df['metric_name'] == metric_name) &
                    (validation_df[row_var] == row_value)
                ].copy()
                
                # Get statistical results
                stat_row = trend_stats[
                    (trend_stats['metric_name'] == metric_name) &
                    (trend_stats[row_var] == row_value)
                ]
                
                # Calculate mean and SEM per continuity level
                grouped = subset.groupby('continuity_setting')['metric_score']
                mean_scores = grouped.mean()
                sem_scores = grouped.sem()
                continuity_levels = mean_scores.index.values
                
                # Plot mean with error shading (using selected main color)
                ax.plot(continuity_levels, mean_scores.values, 
                    'o-', linewidth=2.5, color=colors['main'], 
                    label='Mean', markersize=6, zorder=2)
                
                # Add error bars
                if not sem_scores.isna().all():
                    ax.fill_between(continuity_levels,
                                mean_scores.values - sem_scores.values,
                                mean_scores.values + sem_scores.values,
                                alpha=colors['alpha'], color=colors['main'], zorder=1)
                
                # Add regression line if significant (using selected fit color)
                if len(stat_row) > 0:
                    row = stat_row.iloc[0]
                    
                    if row['regression_p'] < 0.05:
                        x_fit = np.linspace(continuity_levels.min(), continuity_levels.max(), 100)
                        y_fit = row['regression_slope'] * x_fit + (mean_scores.mean() - row['regression_slope'] * continuity_levels.mean())
                        ax.plot(x_fit, y_fit, '--', color=colors['fit'], linewidth=1.5, 
                            alpha=0.7, label='Linear fit', zorder=3)
                    
                    # Significance markers
                    if row['spearman_p'] < 0.001:
                        sig_marker = '***'
                    elif row['spearman_p'] < 0.01:
                        sig_marker = '**'
                    elif row['spearman_p'] < 0.05:
                        sig_marker = '*'
                    else:
                        sig_marker = 'n.s.'
                    
                    # Statistics text
                    stats_text = (
                        f"ρ = {row['spearman_r']:.3f} {sig_marker}\n"
                        f"R² = {row['regression_r2']:.3f}\n"
                        f"Slope = {row['regression_slope']:.3f}"
                    )
                    
                    ax.text(0.05, 0.95, stats_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        fontsize=stats_fontsize)
                
                # Conditionally set labels
                if flat_idx in xlabel_indices:
                    ax.set_xlabel(xlabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax.set_xlabel('')
                
                if flat_idx in ylabel_indices:
                    ylabel_text = f'Metric Score\n({row_label_format(row_value)})'
                    ax.set_ylabel(ylabel_text, fontsize=label_fontsize, fontweight=label_fontweight)
                else:
                    ax.set_ylabel('')
                
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
    
        return fig, axes

    def visualize_two_group_comparison(self,
                                       validation_df: pd.DataFrame,
                                       plot_type: str = 'box',
                                       test_type: str = 'mann-whitney',
                                       figsize: Optional[Tuple[int, int]] = None,
                                       save_path: Optional[str] = None,
                                       show_points: bool = True,
                                       title_fontsize: int = 11,
                                       label_fontsize: int = 9,
                                       tick_fontsize: int = 9,
                                       wspace: float = 0.3,
                                       hspace: float = 0.3,
                                       short_title: Optional[dict] = None,
                                       show_spines: bool = False,
                                       hatch_patterns: Optional[list] = None,
                                       hatch_color: str = 'w',
                                       bar_colors: Optional[list] = None,
                                       sig_fontsize: int = 10,
                                       sig_y_offset: float = 0.05,
                                       sig_bar_height: float = 0.015,
                                       sig_text_offset: float = 0.005):
        """
        Visualize comparison between two continuity levels with statistical tests.
        Works with flexible trajectory/cluster configurations.
        
        Parameters:
            validation_df: DataFrame from validation
            [Other parameters as before]
        """
        # Check for exactly 2 continuity levels
        continuity_levels = validation_df['continuity_setting'].unique()
        if len(continuity_levels) != 2:
            raise ValueError(f"Expected 2 continuity levels, got {len(continuity_levels)}")
        
        # Get unique metrics
        metrics = validation_df['metric_name'].unique()
        
        # Determine row grouping
        if 'config_label' in validation_df.columns:
            row_var = 'config_label'
            row_values = validation_df[row_var].unique()
            row_label_format = lambda x: x
        elif 'n_clusters' in validation_df.columns:
            row_var = 'n_clusters'
            row_values = sorted(validation_df[row_var].unique())
            row_label_format = lambda x: f'{x} Clusters'
        else:
            row_var = 'trajectory_type'
            row_values = validation_df[row_var].unique()
            row_label_format = lambda x: x.capitalize()
        
        # Default short titles
        default_short_titles = {
            'Spectral Decay': 'Spec. Decay',
            'Anisotropy': 'Aniso.',
            'Participation Ratio': 'Part. Ratio',
            'Trajectory Directionality': 'Traj. Direct.',
            'Manifold Dimensionality': 'Manif. Dim.',
            'Noise resilience': 'Noise Resil.'
        }
        
        if short_title is None:
            short_title = default_short_titles
        else:
            short_title = {**default_short_titles, **short_title}
        
        # Defaults
        if hatch_patterns is None:
            hatch_patterns = ['/', '/']
        if bar_colors is None:
            bar_colors = ['lightblue', 'lightcoral']
        
        # Layout
        n_rows = len(row_values)
        n_cols = len(metrics)
        
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        
        stats_results = []
        
        # Plot for each condition
        for row_idx, row_val in enumerate(row_values):
            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                
                # Filter data
                mask = (validation_df['metric_name'] == metric) & (validation_df[row_var] == row_val)
                data = validation_df[mask]
                
                # Prepare data for plotting
                group_0 = data[data['continuity_setting'] == continuity_levels[0]]['metric_score'].values
                group_1 = data[data['continuity_setting'] == continuity_levels[1]]['metric_score'].values
                
                # Remove NaN
                group_0 = group_0[~np.isnan(group_0)]
                group_1 = group_1[~np.isnan(group_1)]
                
                # Statistical test
                if test_type == 'mann-whitney':
                    stat, pval = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                else:
                    stat, pval = stats.ttest_ind(group_0, group_1)
                    test_name = 't-test'
                
                # Store results
                stats_results.append({
                    'metric': metric,
                    row_var: row_val,
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
                    
                    for patch, color in zip(bp['boxes'], bar_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
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
                    
                    for bar, hatch in zip(bars, hatch_patterns):
                        bar.set_hatch(hatch)
                        bar.set_edgecolor(hatch_color)
                    
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f'{continuity_levels[0]:.1f}', f'{continuity_levels[1]:.1f}'])
                    
                    if show_points:
                        for i, group in enumerate([group_0, group_1]):
                            x = np.random.normal(x_pos[i], 0.04, size=len(group))
                            ax.scatter(x, group, alpha=0.4, s=20, c='black', zorder=3)
                
                # Significance annotation
                y_max = max(np.max(group_0), np.max(group_1))
                y_min = min(np.min(group_0), np.min(group_1))
                y_range = y_max - y_min
                
                if pval < 0.001:
                    sig_text = '***'
                elif pval < 0.01:
                    sig_text = '**'
                elif pval < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'n.s.'
                
                # Draw significance bar
                x1, x2 = (1, 2) if plot_type == 'box' else (0, 1)
                y_sig = y_max + y_range * sig_y_offset
                y_bar_top = y_sig + sig_bar_height
                ax.plot([x1, x1, x2, x2], 
                       [y_sig, y_bar_top, y_bar_top, y_sig], 
                       'k-', linewidth=1)
                ax.text((x1 + x2) / 2, y_bar_top + sig_text_offset, sig_text, 
                       ha='center', va='bottom', fontsize=sig_fontsize)
                
                # Labels
                if row_idx == 0:
                    title_text = short_title.get(metric, metric)
                    ax.set_title(title_text, fontsize=title_fontsize)
                
                if col_idx == 0:
                    ax.set_ylabel(row_label_format(row_val), fontsize=label_fontsize)
                
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Continuity', fontsize=label_fontsize)
                
                if not show_spines:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                
                ax.tick_params(labelsize=tick_fontsize)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, axes, pd.DataFrame(stats_results)
    
    # ==================== Helper Methods ====================
    
    def _generate_trajectory(self, traj_type: str, continuity: float, params: Dict) -> Dict:
        """Generate trajectory with type-specific parameters"""
        params = {**params, 'continuity': continuity, 'return_metadata': True}
        
        if traj_type == 'linear':
            return self.simulator.simulate_linear_trajectory(**params)
        elif traj_type == 'branching':
            return self.simulator.simulate_branching_trajectory(**params)
        elif traj_type == 'cyclic':
            return self.simulator.simulate_cyclic_trajectory(**params)
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")


# 1. Flexible Trajectory Configuration
# python
# trajectory_configs = [
#     {'type': 'linear', 'params': {'noise_type': 'gaussian'}},
#     {'type': 'branching', 'params': {'n_branches': 2, 'branch_point': 0.5}},
#     {'type': 'branching', 'params': {'n_branches': 3, 'branch_point': 0.5}},
#     {'type': 'branching', 'params': {'n_branches': 4, 'branch_point': 0.5}},
# ]
# 2. Flexible Cluster Configuration
# python
# cluster_configs = [
#     {'n_clusters': 5, 'target_trajectory': 'linear'},
#     {'n_clusters': 5, 'target_trajectory': 'branching', 'params': {'branch_point': 0.3}},
#     {'n_clusters': 5, 'target_trajectory': 'branching', 'params': {'branch_point': 0.7}},
# ]