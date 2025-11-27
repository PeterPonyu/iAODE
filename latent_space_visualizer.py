from typing import Dict, List, Optional, Tuple
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
                                    trajectory_types: List[str] = ['linear', 'branching', 'cyclic'],
                                    method: str = 'umap',
                                    color_by: str = 'pseudotime',
                                    n_cells: int = 500,
                                    n_dims: int = 50,
                                    # Trajectory-specific parameters
                                    linear_params: Optional[Dict] = None,
                                    branching_params: Optional[Dict] = None,
                                    cyclic_params: Optional[Dict] = None,
                                    # Visualization parameters
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
        Visualize how trajectory structure changes with continuity levels
        
        Parameters:
            simulator: LatentSpaceSimulator instance
            continuity_range: Array of continuity values to visualize
            trajectory_types: List of trajectory types to compare
            method: Embedding method - 'pca', 'umap', or 'tsne'
            color_by: What to color points by ('pseudotime', 'cell_types', etc.)
            n_cells: Default number of cells to simulate (can be overridden per trajectory)
            n_dims: Default dimensionality of latent space (can be overridden per trajectory)
            
            linear_params: Dict of parameters for linear trajectory
                - n_cells: int (default: uses global n_cells)
                - n_dims: int (default: uses global n_dims)
                - noise_type: str (default: 'gaussian')
                Options: 'gaussian', 'uniform', 'heavy_tail'
                
            branching_params: Dict of parameters for branching trajectory
                - n_cells: int (default: uses global n_cells)
                - n_dims: int (default: uses global n_dims)
                - branch_point: float (default: 0.4)
                - n_branches: int (default: 2)
                - branch_angle: float (default: 60)
                
            cyclic_params: Dict of parameters for cyclic trajectory
                - n_cells: int (default: uses global n_cells)
                - n_dims: int (default: uses global n_dims)
                - n_cycles: float (default: 1.5)
                
            save_path: Path to save figure
            figsize: Figure size as (width, height). Default: (5*n_continuity, 5*n_traj)
            title_fontsize: Font size for subplot titles
            title_fontweight: Font weight for subplot titles
            label_fontsize: Font size for axis labels
            label_fontweight: Font weight for axis labels
            tick_fontsize: Font size for axis ticks
            hspace: Height space between subplots
            wspace: Width space between subplots
            show_xlabel: Whether to show x-axis labels
            show_ylabel: Whether to show y-axis labels
            show_xticklabels: Whether to show x-axis tick labels
            show_yticklabels: Whether to show y-axis tick labels
            colorbar_labelsize: Font size for colorbar label
            colorbar_ticksize: Font size for colorbar tick labels
            colorbar_width: Width of colorbar as percentage of axes width
            colorbar_pad: Padding between axes and colorbar
            
        Returns:
            fig, axes: Matplotlib figure and axes objects
            
        Example:
            >>> visualizer.visualize_continuity_range(
            ...     simulator=sim,
            ...     continuity_range=np.linspace(0.8, 0.95, 4),
            ...     trajectory_types=['linear', 'branching', 'cyclic'],
            ...     linear_params={'noise_type': 'heavy_tail'},
            ...     branching_params={'branch_point': 0.5, 'n_branches': 3, 'branch_angle': 45},
            ...     cyclic_params={'n_cycles': 2.0}
            ... )
        """
        # Set default parameters for each trajectory type
        default_linear_params = {
            'n_cells': n_cells,
            'n_dims': n_dims,
            'noise_type': 'gaussian'
        }
        
        default_branching_params = {
            'n_cells': n_cells,
            'n_dims': n_dims,
            'branch_point': 0.4,
            'n_branches': 2,
            'branch_angle': 60
        }
        
        default_cyclic_params = {
            'n_cells': n_cells,
            'n_dims': n_dims,
            'n_cycles': 1.5
        }
        
        # Merge user-provided parameters with defaults
        linear_params = {**default_linear_params, **(linear_params or {})}
        branching_params = {**default_branching_params, **(branching_params or {})}
        cyclic_params = {**default_cyclic_params, **(cyclic_params or {})}
        
        # Create trajectory configuration mapping
        trajectory_configs = {
            'linear': linear_params,
            'branching': branching_params,
            'cyclic': cyclic_params
        }
        
        n_continuity = len(continuity_range)
        n_traj = len(trajectory_types)
        
        # Set default figure size if not provided
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
        
        # Adjust subplot spacing
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        
        # Generate and visualize for each combination
        for i, traj_type in enumerate(trajectory_types):
            # Get trajectory-specific parameters
            traj_params = trajectory_configs.get(traj_type, {})
            
            for j, continuity in enumerate(continuity_range):
                ax = axes[i, j]
                
                # Generate simulated data with trajectory-specific parameters
                sim_data = self._generate_trajectory(
                    simulator=simulator,
                    traj_type=traj_type,
                    continuity=continuity,
                    params=traj_params
                )
                
                latent_space = sim_data['latent_space']
                metadata = sim_data
                
                # Compute embedding
                if method.lower() == 'pca':
                    embedding = self._compute_pca(latent_space)
                elif method.lower() == 'tsne':
                    embedding = self._compute_tsne(latent_space)
                elif method.lower() == 'umap':
                    embedding = self._compute_umap(latent_space)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Get coloring information
                color_data, color_label, cmap, is_continuous = self._get_color_info(
                    metadata, color_by
                )
                
                # Plot with title showing continuity level
                title = f"Continuity = {continuity:.2f}"
                
                # Determine if this subplot should show labels
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
                    row_label=f'{traj_type.capitalize()}' if j == 0 else '',
                    colorbar_labelsize=colorbar_labelsize,
                    colorbar_ticksize=colorbar_ticksize,
                    colorbar_width=colorbar_width,
                    colorbar_pad=colorbar_pad
                )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig, axes

    def visualize_cluster_continuity_range(self,
                                          simulator,
                                          continuity_range: np.ndarray = np.linspace(0.0, 1.0, 6),
                                          n_clusters_list: List[int] = [3, 5, 8],
                                          target_trajectory: str = 'linear',
                                          base_separation: float = 5.0,
                                          branch_point: float = 0.25,
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
        Visualize how cluster structure changes with continuity levels
        
        Parameters:
            simulator: LatentSpaceSimulator instance
            continuity_range: Array of continuity values (0.0 to 1.0)
            n_clusters_list: List of cluster numbers to compare
            target_trajectory: 'linear', 'branching', or 'cyclic'
            base_separation: Base distance between cluster centers
            method: Embedding method - 'pca', 'umap', or 'tsne'
            color_by: 'cluster_labels' or 'cell_types'
            n_cells: Number of cells to simulate
            n_dims: Dimensionality of latent space
            save_path: Path to save figure
            figsize: Figure size as (width, height)
            title_fontsize: Font size for subplot titles
            title_fontweight: Font weight for titles
            label_fontsize: Font size for axis labels
            label_fontweight: Font weight for labels
            tick_fontsize: Font size for tick labels
            hspace: Height space between subplots
            wspace: Width space between subplots
            show_xlabel: Whether to show x-axis labels
            show_ylabel: Whether to show y-axis labels
            show_xticklabels: Whether to show x-axis tick labels
            show_yticklabels: Whether to show y-axis tick labels
            show_legend: Whether to show legend for clusters
            legend_fontsize: Font size for legend
        """
        n_continuity = len(continuity_range)
        n_cluster_settings = len(n_clusters_list)
        
        if figsize is None:
            figsize = (5*n_continuity, 5*n_cluster_settings)
        
        fig, axes = plt.subplots(n_cluster_settings, n_continuity, figsize=figsize)
        
        # Handle single row/column cases
        if n_cluster_settings == 1 and n_continuity == 1:
            axes = np.array([[axes]])
        elif n_cluster_settings == 1:
            axes = axes.reshape(1, -1)
        elif n_continuity == 1:
            axes = axes.reshape(-1, 1)
        
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        
        # Generate colormap for clusters
        cmap = plt.cm.tab10
        
        for i, n_clusters in enumerate(n_clusters_list):
            for j, continuity in enumerate(continuity_range):
                ax = axes[i, j]
                
                # Generate data
                sim_data = simulator.simulate_discrete_clusters(
                    n_cells=n_cells,
                    n_clusters=n_clusters,
                    n_dims=n_dims,
                    continuity=continuity,
                    base_separation=base_separation,
                    target_trajectory=target_trajectory,
                    branch_point=branch_point,
                    return_metadata=True
                )
                
                latent_space = sim_data['latent_space']
                cluster_labels = sim_data['cluster_labels']
                
                # Compute embedding
                if method.lower() == 'pca':
                    embedding = self._compute_pca(latent_space)
                elif method.lower() == 'tsne':
                    embedding = self._compute_tsne(latent_space)
                elif method.lower() == 'umap':
                    embedding = self._compute_umap(latent_space)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Plot each cluster
                for cluster_id in range(n_clusters):
                    mask = cluster_labels == cluster_id
                    ax.scatter(embedding[mask, 0], embedding[mask, 1],
                              c=[cmap(cluster_id % 10)], 
                              s=20, alpha=0.6,
                              label=f'C{cluster_id}')
                
                # Title
                ax.set_title(f"Continuity = {continuity:.2f}", 
                            fontsize=title_fontsize, fontweight=title_fontweight)
                
                # X label (bottom row only)
                if show_xlabel and i == n_cluster_settings - 1:
                    ax.set_xlabel(f'{method.upper()} 1', fontsize=label_fontsize, fontweight=label_fontweight)
                
                # Y label (left column only, with row info)
                if show_ylabel and j == 0:
                    ax.set_ylabel(f'{n_clusters} Clusters', 
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig, axes
    
    # ==================== Helper Methods ====================

    
    def _generate_trajectory(self, 
                            simulator, 
                            traj_type: str, 
                            continuity: float,
                            params: Dict) -> Dict:
        """
        Helper method to generate trajectory with type-specific parameters
        
        Parameters:
            simulator: LatentSpaceSimulator instance
            traj_type: Type of trajectory ('linear', 'branching', 'cyclic')
            continuity: Continuity value for this simulation
            params: Dictionary of trajectory-specific parameters
            
        Returns:
            Dictionary containing simulated data and metadata
        """
        # Extract common parameters
        n_cells = params.get('n_cells', 500)
        n_dims = params.get('n_dims', 50)
        
        if traj_type == 'linear':
            sim_data = simulator.simulate_linear_trajectory(
                n_cells=n_cells,
                n_dims=n_dims,
                continuity=continuity,
                noise_type=params.get('noise_type', 'gaussian'),
                return_metadata=True
            )
            
        elif traj_type == 'branching':
            sim_data = simulator.simulate_branching_trajectory(
                n_cells=n_cells,
                n_dims=n_dims,
                continuity=continuity,
                branch_point=params.get('branch_point', 0.4),
                n_branches=params.get('n_branches', 2),
                branch_angle=params.get('branch_angle', 60),
                return_metadata=True
            )
            
        elif traj_type == 'cyclic':
            sim_data = simulator.simulate_cyclic_trajectory(
                n_cells=n_cells,
                n_dims=n_dims,
                continuity=continuity,
                n_cycles=params.get('n_cycles', 1.5),
                return_metadata=True
            )
            
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
        
        return sim_data

    
    def _compute_pca(self, data, n_components=2):
        """Compute PCA embedding"""
        from sklearn.decomposition import PCA
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
            return (metadata.get('pseudotime', np.arange(len(metadata))), 
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
        """
        Helper method to plot 2D embeddings
        
        Parameters:
            ax: Matplotlib axis
            embedding: 2D embedding coordinates
            color_data: Data for coloring points
            cmap: Colormap
            is_continuous: Whether color_data is continuous
            title: Plot title
            color_label: Label for colorbar/legend
            title_fontsize: Font size for title
            title_fontweight: Font weight for title
            label_fontsize: Font size for axis labels
            label_fontweight: Font weight for axis labels
            tick_fontsize: Font size for ticks
            show_xlabel: Whether to show x-axis label
            show_ylabel: Whether to show y-axis label
            show_xticklabels: Whether to show x-axis tick labels
            show_yticklabels: Whether to show y-axis tick labels
            row_label: Optional label for the row (trajectory type)
            colorbar_labelsize: Font size for colorbar label
            colorbar_ticksize: Font size for colorbar tick labels
            colorbar_width: Width of colorbar as percentage of axes width
            colorbar_pad: Padding between axes and colorbar
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        if is_continuous:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                               c=color_data, cmap=cmap, s=20, alpha=0.7)
            if color_label:
                # Create divider for existing axes
                divider = make_axes_locatable(ax)
                # Append axes to the right of ax, with specified width and padding
                cax = divider.append_axes("right", size=colorbar_width, pad=colorbar_pad)
                # Create colorbar in the new axes
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
                legend = ax.legend(title=color_label, fontsize=tick_fontsize,
                                  title_fontsize=colorbar_labelsize)
        
        # Set labels conditionally
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
        
        # Set tick labels conditionally
        if not show_xticklabels:
            ax.tick_params(axis='x', labelbottom=False, bottom=False)
        else:
            ax.tick_params(axis='x', labelsize=tick_fontsize)
        
        if not show_yticklabels:
            ax.tick_params(axis='y', labelleft=False, left=False)
        else:
            ax.tick_params(axis='y', labelsize=tick_fontsize)
        
        ax.grid(True, alpha=0.3, linestyle='--')

