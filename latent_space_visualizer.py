from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class LatentSpaceVisualizer:
    """
    Comprehensive 2D visualization tools for single-cell latent spaces
    """
    
    def __init__(self, figsize=(20, 5)):
        self.figsize = figsize
        self.colors = {
            'trajectory': 'viridis',
            'cell_type': 'tab10',
            'branch': 'Set2',
            'cluster': 'tab20'
        }
        
    def visualize_continuity_range(self,
                                    simulator,
                                    continuity_range: np.ndarray = np.linspace(0.9, 0.99, 9),
                                    trajectory_configs: Optional[List[Dict]] = None,
                                    method: str = 'umap',
                                    color_by: str = 'pseudotime',
                                    n_cells: int = 500,
                                    n_dims: int = 50,
                                    save_path: Optional[str] = None,
                                    figsize: Optional[Tuple[int, int]] = None,
                                    title_fontsize: int = 12,
                                    title_fontweight: str = 'normal',
                                    label_fontsize: int = 9,
                                    label_fontweight: str = 'normal',
                                    tick_fontsize: int = 9,
                                    hspace: float = 0.3,
                                    wspace: float = 0.3,
                                    show_xlabel: bool = True,
                                    show_ylabel: bool = True,
                                    show_xticklabels: bool = True,
                                    show_yticklabels: bool = True,
                                    colorbar_labelsize: int = 9,
                                    colorbar_ticksize: int = 9,
                                    colorbar_width: str = '5%',
                                    colorbar_pad: float = 0.1):
        """
        Visualize how trajectory structure changes with continuity levels.
        
        Supports flexible trajectory configuration - multiple instances of same type with different parameters.
        
        Parameters:
            simulator: LatentSpaceSimulator instance
            continuity_range: Array of continuity values to visualize
            trajectory_configs: List of trajectory configurations, each containing:
                - 'type': str - 'linear', 'branching', or 'cyclic'
                - 'params': dict - trajectory-specific parameters
                - 'label': str (optional) - custom label for row
                
                Example:
                    [
                        {'type': 'linear', 'params': {'noise_type': 'gaussian'}},
                        {'type': 'branching', 'params': {'n_branches': 2, 'branch_point': 0.5}},
                        {'type': 'branching', 'params': {'n_branches': 3, 'branch_point': 0.5}, 'label': 'Branching 3-way'},
                    ]
            
            method: Embedding method - 'pca', 'umap', or 'tsne'
            color_by: What to color points by ('pseudotime', 'cell_types', etc.)
            n_cells: Default number of cells (overridden by params['n_cells'])
            n_dims: Default dimensionality (overridden by params['n_dims'])
            
            save_path: Path to save figure
            figsize: Figure size (width, height). Default: auto-scaled
            [Font/layout parameters as before]
            
        Returns:
            fig, axes: Matplotlib figure and axes objects
        """
        # Default trajectory configs if none provided
        if trajectory_configs is None:
            trajectory_configs = [
                {'type': 'linear', 'params': {}},
                {'type': 'branching', 'params': {}},
                {'type': 'cyclic', 'params': {}},
            ]
        
        # Standardize configs and add defaults
        standardized_configs = []
        for config in trajectory_configs:
            std_config = {
                'type': config['type'],
                'params': {'n_cells': n_cells, 'n_dims': n_dims, **config.get('params', {})},
                'label': config.get('label', config['type'].capitalize())
            }
            standardized_configs.append(std_config)
        
        n_continuity = len(continuity_range)
        n_traj = len(standardized_configs)
        
        # Set default figure size
        if figsize is None:
            figsize = (5*n_continuity, 5*n_traj)
        
        fig, axes = plt.subplots(n_traj, n_continuity, figsize=figsize)
        
        # Handle single row/column cases
        if n_traj == 1 and n_continuity == 1:
            axes = np.array([[axes]])
        elif n_traj == 1:
            axes = axes.reshape(1, -1)
        elif n_continuity == 1:
            axes = axes.reshape(-1, 1)
        
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        
        # Generate and visualize for each combination
        for i, config in enumerate(standardized_configs):
            traj_type = config['type']
            traj_params = config['params']
            traj_label = config['label']
            
            for j, continuity in enumerate(continuity_range):
                ax = axes[i, j]
                
                # Generate simulated data
                sim_data = self._generate_trajectory(
                    simulator=simulator,
                    traj_type=traj_type,
                    continuity=continuity,
                    params=traj_params
                )
                
                latent_space = sim_data['latent_space']
                metadata = sim_data
                
                # Compute embedding
                embedding = self._compute_embedding(latent_space, method)
                
                # Get coloring information
                color_data, color_label, cmap, is_continuous = self._get_color_info(
                    metadata, color_by
                )
                
                # Plot
                title = f"Continuity = {continuity:.2f}"
                show_xlabel_this = show_xlabel and (i == n_traj - 1)
                show_ylabel_this = show_ylabel and (j == 0)
                
                self._plot_embedding(
                    ax, embedding, color_data, cmap, is_continuous,
                    title=title, color_label=color_label if j == n_continuity-1 else '',
                    title_fontsize=title_fontsize, title_fontweight=title_fontweight,
                    label_fontsize=label_fontsize, label_fontweight=label_fontweight,
                    tick_fontsize=tick_fontsize,
                    show_xlabel=show_xlabel_this,
                    show_ylabel=show_ylabel_this,
                    show_xticklabels=show_xticklabels,
                    show_yticklabels=show_yticklabels,
                    row_label=traj_label if j == 0 else '',
                    colorbar_labelsize=colorbar_labelsize,
                    colorbar_ticksize=colorbar_ticksize,
                    colorbar_width=colorbar_width,
                    colorbar_pad=colorbar_pad
                )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, axes

    def visualize_cluster_continuity_range(self,
                                          simulator,
                                          continuity_range: np.ndarray = np.linspace(0.0, 1.0, 6),
                                          cluster_configs: Optional[List[Dict]] = None,
                                          method: str = 'umap',
                                          color_by: str = 'cluster_labels',
                                          n_cells: int = 500,
                                          n_dims: int = 50,
                                          save_path: Optional[str] = None,
                                          figsize: Optional[Tuple[int, int]] = None,
                                          title_fontsize: int = 12,
                                          title_fontweight: str = 'normal',
                                          label_fontsize: int = 9,
                                          label_fontweight: str = 'normal',
                                          tick_fontsize: int = 9,
                                          hspace: float = 0.3,
                                          wspace: float = 0.3,
                                          show_xlabel: bool = True,
                                          show_ylabel: bool = True,
                                          show_xticklabels: bool = True,
                                          show_yticklabels: bool = True,
                                          show_legend: bool = True,
                                          legend_fontsize: int = 8):
        """
        Visualize how cluster structure changes with continuity levels.
        
        Supports flexible cluster configuration - multiple instances with different parameters.
        
        Parameters:
            simulator: LatentSpaceSimulator instance
            continuity_range: Array of continuity values (0.0 to 1.0)
            cluster_configs: List of cluster configurations, each containing:
                - 'n_clusters': int - number of clusters
                - 'target_trajectory': str - 'linear', 'branching', or 'cyclic'
                - 'params': dict (optional) - additional parameters
                - 'label': str (optional) - custom label for row
                
                Example:
                    [
                        {'n_clusters': 3, 'target_trajectory': 'linear'},
                        {'n_clusters': 5, 'target_trajectory': 'branching', 'params': {'branch_point': 0.3}},
                        {'n_clusters': 5, 'target_trajectory': 'branching', 'params': {'branch_point': 0.6}},
                    ]
            
            method: Embedding method - 'pca', 'umap', or 'tsne'
            color_by: 'cluster_labels' or 'cell_types'
            n_cells: Default number of cells
            n_dims: Default dimensionality
            [Other parameters as before]
        """
        # Default cluster configs if none provided
        if cluster_configs is None:
            cluster_configs = [
                {'n_clusters': 3, 'target_trajectory': 'linear'},
                {'n_clusters': 5, 'target_trajectory': 'linear'},
                {'n_clusters': 8, 'target_trajectory': 'linear'},
            ]
        
        # Standardize configs
        standardized_configs = []
        for config in cluster_configs:
            n_clusters = config['n_clusters']
            target_traj = config['target_trajectory']
            params = config.get('params', {})
            
            # Default parameters for simulate_discrete_clusters
            default_params = {
                'n_cells': n_cells,
                'n_dims': n_dims,
                'base_separation': 5.0,
                'branch_point': 0.5,
            }
            merged_params = {**default_params, **params}
            
            # Generate label
            if 'label' in config:
                label = config['label']
            else:
                if params:
                    param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
                    label = f"{n_clusters} Clusters ({target_traj}, {param_str})"
                else:
                    label = f"{n_clusters} Clusters ({target_traj})"
            
            std_config = {
                'n_clusters': n_clusters,
                'target_trajectory': target_traj,
                'params': merged_params,
                'label': label
            }
            standardized_configs.append(std_config)
        
        n_continuity = len(continuity_range)
        n_configs = len(standardized_configs)
        
        if figsize is None:
            figsize = (5*n_continuity, 5*n_configs)
        
        fig, axes = plt.subplots(n_configs, n_continuity, figsize=figsize)
        
        # Handle single row/column cases
        if n_configs == 1 and n_continuity == 1:
            axes = np.array([[axes]])
        elif n_configs == 1:
            axes = axes.reshape(1, -1)
        elif n_continuity == 1:
            axes = axes.reshape(-1, 1)
        
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        
        # Generate colormap for clusters (support up to 20 clusters)
        cmap = plt.cm.tab20
        
        for i, config in enumerate(standardized_configs):
            n_clusters = config['n_clusters']
            target_traj = config['target_trajectory']
            params = config['params']
            config_label = config['label']
            
            for j, continuity in enumerate(continuity_range):
                ax = axes[i, j]
                
                # Generate data
                sim_data = simulator.simulate_discrete_clusters(
                    n_clusters=n_clusters,
                    continuity=continuity,
                    target_trajectory=target_traj,
                    return_metadata=True,
                    **params
                )
                
                latent_space = sim_data['latent_space']
                cluster_labels = sim_data['cluster_labels']
                
                # Compute embedding
                embedding = self._compute_embedding(latent_space, method)
                
                # Plot each cluster
                for cluster_id in range(n_clusters):
                    mask = cluster_labels == cluster_id
                    ax.scatter(embedding[mask, 0], embedding[mask, 1],
                              c=[cmap(cluster_id % 20)], 
                              s=20, alpha=0.6,
                              label=f'C{cluster_id}')
                
                # Title
                ax.set_title(f"Continuity = {continuity:.2f}", 
                            fontsize=title_fontsize, fontweight=title_fontweight)
                
                # X label (bottom row only)
                if show_xlabel and i == n_configs - 1:
                    ax.set_xlabel(f'{method.upper()} 1', 
                                 fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Y label (left column only, with row info)
                if show_ylabel and j == 0:
                    ax.set_ylabel(config_label, 
                                 fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Tick labels
                if not show_xticklabels:
                    ax.tick_params(labelbottom=False, bottom=False)
                if not show_yticklabels:
                    ax.tick_params(labelleft=False, left=False)
                
                ax.tick_params(labelsize=tick_fontsize)
                
                # Legend (rightmost plots only)
                if show_legend and j == n_continuity - 1:
                    ax.legend(fontsize=legend_fontsize, loc='center left', 
                             bbox_to_anchor=(1, 0.5), frameon=False)
                
                ax.grid(True, alpha=0.3, linestyle='--')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
        return fig, axes
    
    # ==================== Helper Methods ====================
    
    def _generate_trajectory(self, 
                            simulator, 
                            traj_type: str, 
                            continuity: float,
                            params: Dict) -> Dict:
        """
        Generate trajectory with type-specific parameters.
        
        Parameters:
            simulator: LatentSpaceSimulator instance
            traj_type: 'linear', 'branching', or 'cyclic'
            continuity: Continuity value
            params: Trajectory-specific parameters
        """
        # Add continuity to params
        params = {**params, 'continuity': continuity, 'return_metadata': True}
        
        if traj_type == 'linear':
            return simulator.simulate_linear_trajectory(**params)
            
        elif traj_type == 'branching':
            return simulator.simulate_branching_trajectory(**params)
            
        elif traj_type == 'cyclic':
            return simulator.simulate_cyclic_trajectory(**params)
            
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}. Must be 'linear', 'branching', or 'cyclic'")
    
    def _compute_embedding(self, data, method: str, n_components: int = 2):
        """Compute 2D embedding using specified method."""
        if method.lower() == 'pca':
            return self._compute_pca(data, n_components)
        elif method.lower() == 'tsne':
            return self._compute_tsne(data, n_components)
        elif method.lower() == 'umap':
            return self._compute_umap(data, n_components)
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'pca', 'tsne', or 'umap'")
    
    def _compute_pca(self, data, n_components=2):
        """Compute PCA embedding"""
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)
    
    def _compute_tsne(self, data, n_components=2, perplexity=30, random_state=42):
        """Compute t-SNE embedding"""
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   random_state=random_state, n_jobs=-1)
        return tsne.fit_transform(data)
    
    def _compute_umap(self, data, n_components=2, n_neighbors=15, 
                     min_dist=0.1, random_state=42):
        """Compute UMAP embedding"""
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=random_state)
        return reducer.fit_transform(data)
    
    def _get_color_info(self, metadata, color_by):
        """Extract coloring information from metadata"""
        if color_by == 'pseudotime':
            return (metadata.get('pseudotime'), 'Pseudotime', 
                   'viridis', True)
        elif color_by == 'cell_types':
            cell_types = metadata.get('cell_types')
            unique_types = sorted(set(cell_types))
            type_to_num = {t: i for i, t in enumerate(unique_types)}
            color_nums = np.array([type_to_num[t] for t in cell_types])
            return (color_nums, 'Cell Type', 'tab10', False)
        elif color_by == 'branch_id':
            return (metadata.get('branch_id'), 'Branch ID', 'Set2', False)
        elif color_by == 'cluster_labels':
            return (metadata.get('cluster_labels'), 'Cluster', 'tab20', False)
        elif color_by == 'cycle_phase':
            return (metadata.get('cycle_phase'), 'Cell Cycle Phase', 
                   'twilight', True)
        else:
            # Default to pseudotime
            return (metadata.get('pseudotime', np.arange(len(metadata.get('latent_space', [])))), 
                   color_by, 'viridis', True)
    
    def _plot_embedding(self, ax, embedding, color_data, cmap, is_continuous,
                       title: str = '', color_label: str = '',
                       title_fontsize: int = 12, title_fontweight: str = 'normal',
                       label_fontsize: int = 9, label_fontweight: str = 'normal',
                       tick_fontsize: int = 9,
                       show_xlabel: bool = False,
                       show_ylabel: bool = True,
                       show_xticklabels: bool = False,
                       show_yticklabels: bool = False,
                       row_label: str = '',
                       colorbar_labelsize: int = 9,
                       colorbar_ticksize: int = 8,
                       colorbar_width: str = '5%',
                       colorbar_pad: float = 0.1):
        """Plot 2D embeddings with flexible styling"""
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        if is_continuous:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                               c=color_data, cmap=cmap, s=20, alpha=0.7)
            if color_label:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=colorbar_width, pad=colorbar_pad)
                cbar = plt.colorbar(scatter, cax=cax)
                cbar.set_label(color_label, fontsize=colorbar_labelsize)
                cbar.ax.tick_params(labelsize=colorbar_ticksize)
        else:
            # Categorical coloring
            unique_vals = np.unique(color_data)
            for val in unique_vals:
                mask = color_data == val
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          label=f'{val}', s=20, alpha=0.7)
            if color_label:
                ax.legend(title=color_label, fontsize=tick_fontsize,
                         title_fontsize=colorbar_labelsize)
        
        # Labels
        if show_xlabel:
            ax.set_xlabel('Dim 1', fontsize=label_fontsize, fontweight=label_fontweight)
        else:
            ax.set_xlabel('')
        
        if show_ylabel:
            if row_label:
                ax.set_ylabel(f'{row_label}', 
                             fontsize=label_fontsize, fontweight=label_fontweight)
            else:
                ax.set_ylabel('Dim 2', fontsize=label_fontsize, fontweight=label_fontweight)
        else:
            ax.set_ylabel('')
        
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        # Tick labels
        if not show_xticklabels:
            ax.tick_params(axis='x', labelbottom=False, bottom=False)
        else:
            ax.tick_params(axis='x', labelsize=tick_fontsize)
        
        if not show_yticklabels:
            ax.tick_params(axis='y', labelleft=False, left=False)
        else:
            ax.tick_params(axis='y', labelsize=tick_fontsize)
        
        ax.grid(True, alpha=0.3, linestyle='--')



## Example 1: Multiple branching configurations
# visualizer.visualize_continuity_range(
#     simulator=sim,
#     continuity_range=np.linspace(0.9, 0.99, 5),
#     trajectory_configs=[
#         {'type': 'branching', 'params': {'n_branches': 2, 'branch_point': 0.5}},
#         {'type': 'branching', 'params': {'n_branches': 3, 'branch_point': 0.5}},
#         {'type': 'branching', 'params': {'n_branches': 4, 'branch_point': 0.5}},
#     ],
#     method='umap'
# )

# # Example 2: Mixed trajectory types without cyclic
# visualizer.visualize_continuity_range(
#     simulator=sim,
#     continuity_range=np.linspace(0.8, 0.95, 4),
#     trajectory_configs=[
#         {'type': 'linear', 'params': {'noise_type': 'gaussian'}},
#         {'type': 'branching', 'params': {'n_branches': 2, 'branch_point': 0.4}},
#         {'type': 'branching', 'params': {'n_branches': 3, 'branch_point': 0.6}, 'label': '3-way Late'},
#     ]
# )

# # Example 3: Cluster configurations with different target trajectories
# visualizer.visualize_cluster_continuity_range(
#     simulator=sim,
#     continuity_range=np.linspace(0.0, 1.0, 6),
#     cluster_configs=[
#         {'n_clusters': 5, 'target_trajectory': 'linear'},
#         {'n_clusters': 5, 'target_trajectory': 'branching', 'params': {'branch_point': 0.3}},
#         {'n_clusters': 5, 'target_trajectory': 'branching', 'params': {'branch_point': 0.7}},
#         {'n_clusters': 8, 'target_trajectory': 'cyclic'},
#     ]
# )