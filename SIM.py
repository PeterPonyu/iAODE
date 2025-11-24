import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import poisson, nbinom
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional, Callable

import umap
import warnings
warnings.filterwarnings('ignore')

class LatentSpaceSimulator:
    """
    Comprehensive continous simulation framework for single-cell latent spaces
    
    Supports:
    - Multiple trajectory types (linear, branching, cyclic, complex)
    - scATAC-seq specific simulations with realistic noise
    - Ground truth continuity control
    - Multiple noise models
    - Batch effects
    """
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
    # ==================== Basic Trajectory Generators ====================
    
    def simulate_linear_trajectory(self, 
                                   n_cells: int = 1000,
                                   n_dims: int = 50,
                                   continuity: float = 0.9,
                                   noise_type: str = 'gaussian',
                                   return_metadata: bool = True) -> Dict:
        """
        Simulate simple linear developmental trajectory
        
        Parameters:
            n_cells: Number of cells
            n_dims: Dimensionality of latent space
            continuity: 0-1, controls trajectory smoothness
            noise_type: 'gaussian', 'uniform', 'heavy_tail'
            return_metadata: Return ground truth information
            
        Returns:
            Dict with latent_space, pseudotime, and metadata
        """
        # Generate smooth pseudotime
        pseudotime = np.linspace(0, 1, n_cells)
        
        # Primary trajectory (first 3 dimensions)
        trajectory_dims = min(3, n_dims)
        trajectory = np.zeros((n_cells, trajectory_dims))
        
        # Linear progression with smooth variation
        trajectory[:, 0] = pseudotime * continuity
        trajectory[:, 1] = np.sin(2 * np.pi * pseudotime) * continuity * 0.5
        trajectory[:, 2] = np.cos(2 * np.pi * pseudotime) * continuity * 0.3
        
        # Add noise to remaining dimensions
        noise_scale = (1 - continuity)
        noise = self._generate_noise(n_cells, n_dims - trajectory_dims, 
                                     noise_type, scale=noise_scale)
        
        # Combine trajectory and noise
        latent_space = np.column_stack([trajectory, noise])
        
        if return_metadata:
            return {
                'latent_space': latent_space,
                'pseudotime': pseudotime,
                'cell_types': self._assign_cell_types_linear(pseudotime),
                'continuity': continuity,
                'trajectory_type': 'linear',
                'ground_truth': {
                    'effective_dims': trajectory_dims,
                    'noise_dims': n_dims - trajectory_dims,
                    'true_continuity': continuity
                }
            }
        return latent_space
    
    def simulate_branching_trajectory(self,
                                     n_cells: int = 1500,
                                     n_dims: int = 50,
                                     continuity: float = 0.85,
                                     branch_point: float = 0.4,
                                     n_branches: int = 2,
                                     branch_angle: float = 60,
                                     return_metadata: bool = True) -> Dict:
        """
        Simulate branching developmental trajectory (e.g., differentiation)
        
        Parameters:
            branch_point: Position along trajectory where branching occurs (0-1)
            n_branches: Number of branches (2 or 3)
            branch_angle: Angle between branches in degrees
        """
        # Pre-branch cells
        n_pre_branch = int(n_cells * branch_point)
        n_post_branch = n_cells - n_pre_branch
        
        # Pre-branch trajectory
        pre_time = np.linspace(0, branch_point, n_pre_branch)
        pre_trajectory = np.column_stack([
            pre_time * continuity,
            np.sin(2 * np.pi * pre_time) * continuity * 0.3,
            np.zeros(n_pre_branch)
        ])
        
        # Branch assignments
        branch_labels = np.random.choice(n_branches, n_post_branch)
        branch_sizes = [np.sum(branch_labels == i) for i in range(n_branches)]
        
        # Post-branch trajectories
        post_trajectories = []
        branch_times = []
        
        for branch_id in range(n_branches):
            n_branch_cells = branch_sizes[branch_id]
            if n_branch_cells == 0:
                continue
                
            branch_time = np.linspace(branch_point, 1, n_branch_cells)
            branch_times.extend(branch_time)
            
            # Calculate branch direction
            angle_rad = np.radians(branch_angle * (branch_id - (n_branches-1)/2))
            
            branch_traj = np.column_stack([
                branch_time * continuity,
                np.sin(2 * np.pi * branch_time) * continuity * 0.3 + 
                    (branch_time - branch_point) * np.sin(angle_rad) * continuity,
                (branch_time - branch_point) * np.cos(angle_rad) * continuity
            ])
            post_trajectories.append(branch_traj)
        
        # Combine all trajectories
        post_trajectory = np.vstack(post_trajectories)
        full_trajectory = np.vstack([pre_trajectory, post_trajectory])
        
        # Add noise dimensions
        noise_scale = (1 - continuity)
        noise = self._generate_noise(n_cells, n_dims - 3, 'gaussian', scale=noise_scale)
        
        latent_space = np.column_stack([full_trajectory, noise])
        
        # Metadata
        pseudotime = np.concatenate([pre_time, branch_times])
        branch_assignment = np.concatenate([
            np.full(n_pre_branch, -1),  # -1 for pre-branch
            branch_labels
        ])
        
        if return_metadata:
            return {
                'latent_space': latent_space,
                'pseudotime': pseudotime,
                'branch_id': branch_assignment,
                'cell_types': self._assign_cell_types_branching(pseudotime, branch_assignment),
                'continuity': continuity,
                'trajectory_type': 'branching',
                'ground_truth': {
                    'branch_point': branch_point,
                    'n_branches': n_branches,
                    'effective_dims': 3,
                    'true_continuity': continuity
                }
            }
        return latent_space
    
    def simulate_cyclic_trajectory(self,
                                  n_cells: int = 1000,
                                  n_dims: int = 50,
                                  continuity: float = 0.9,
                                  n_cycles: float = 1.5,
                                  return_metadata: bool = True) -> Dict:
        """
        Simulate cyclic trajectory (e.g., cell cycle)
        
        Parameters:
            n_cycles: Number of complete cycles
        """
        # Cyclic pseudotime
        theta = np.linspace(0, 2 * np.pi * n_cycles, n_cells)
        pseudotime = theta / (2 * np.pi * n_cycles)  # Normalize to [0,1]
        
        # Circular trajectory in 2D with 3D spiral
        trajectory = np.column_stack([
            np.cos(theta) * continuity,
            np.sin(theta) * continuity,
            pseudotime * continuity * 0.5  # Slight progression through cycles
        ])
        
        # Add noise
        noise_scale = (1 - continuity)
        noise = self._generate_noise(n_cells, n_dims - 3, 'gaussian', scale=noise_scale)
        
        latent_space = np.column_stack([trajectory, noise])
        
        if return_metadata:
            return {
                'latent_space': latent_space,
                'pseudotime': pseudotime,
                'cycle_phase': theta % (2 * np.pi),
                'cell_types': self._assign_cell_types_cyclic(theta),
                'continuity': continuity,
                'trajectory_type': 'cyclic',
                'ground_truth': {
                    'n_cycles': n_cycles,
                    'effective_dims': 3,
                    'true_continuity': continuity
                }
            }
        return latent_space
    
    
    def simulate_discrete_clusters(self,
                              n_cells: int = 1000,
                              n_clusters: int = 5,
                              n_dims: int = 50,
                              continuity: float = 0.0,
                              base_separation: float = 5.0,
                              target_trajectory: str = 'linear',
                              branch_point: float = 0.5,
                              return_metadata: bool = True) -> Dict:
        """
        Simulate discrete-to-continuous cluster transitions
        
        Smooth transition from discrete clusters to continuous trajectory
        """
        # Noise decreases smoothly with continuity
        cluster_noise = 0.8 * (1.0 - 0.8 * continuity)
        
        # Separation decreases with continuity
        separation = base_separation * (1.0 - 0.3 * continuity)
        
        # Generate cluster centers based on target trajectory
        cluster_centers = np.zeros((n_clusters, n_dims))
        
        if target_trajectory == 'linear':
            positions = np.linspace(0, 1, n_clusters)
            cluster_centers[:, 0] = positions * separation * 2 - separation
            cluster_centers[:, 1] = 0
            
        elif target_trajectory == 'branching':
            # Calculate cluster allocation
            n_trunk = max(1, int(n_clusters * branch_point))
            n_remaining = n_clusters - n_trunk
            n_branch_a = (n_remaining + 1) // 2
            n_branch_b = n_remaining - n_branch_a
            
            total_length = separation * 2
            trunk_length = branch_point * total_length
            branch_length = (1 - branch_point) * total_length
            
            # === TRUNK: Linear progression to branch point ===
            for i in range(n_trunk):
                t = i / max(1, n_trunk - 1)
                cluster_centers[i, 0] = t * trunk_length
                cluster_centers[i, 1] = 0
            
            # Branch point coordinates (last trunk cluster position)
            branch_x = cluster_centers[n_trunk - 1, 0]
            branch_y = cluster_centers[n_trunk - 1, 1]
            
            # Branch angles
            angle_a = np.radians(45)
            angle_b = np.radians(-45)
            
            # === BRANCH A (upper): Start FROM branch point ===
            for i in range(n_branch_a):
                # KEY FIX: Start at i/(n_branch_a), so first cluster is closer to branch point
                t = i / max(1, n_branch_a - 1) if n_branch_a > 1 else 0
                dist = t * branch_length
                
                idx = n_trunk + i
                cluster_centers[idx, 0] = branch_x + dist * np.cos(angle_a)
                cluster_centers[idx, 1] = branch_y + dist * np.sin(angle_a)
            
            # === BRANCH B (lower): Start FROM branch point ===
            for i in range(n_branch_b):
                # KEY FIX: Start at i/(n_branch_b), so first cluster is closer to branch point
                t = i / max(1, n_branch_b - 1) if n_branch_b > 1 else 0
                dist = t * branch_length
                
                idx = n_trunk + n_branch_a + i
                cluster_centers[idx, 0] = branch_x + dist * np.cos(angle_b)
                cluster_centers[idx, 1] = branch_y + dist * np.sin(angle_b)
        
        elif target_trajectory == 'cyclic':
            angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
            cluster_centers[:, 0] = separation * np.cos(angles)
            cluster_centers[:, 1] = separation * np.sin(angles)
        
        else:
            raise ValueError(f"target_trajectory must be 'linear', 'branching', or 'cyclic'")
        
        # Higher dimensions
        if n_dims > 2:
            cluster_centers[:, 2:] = np.random.randn(n_clusters, n_dims - 2) * \
                                     separation * 0.2 * (1 - 0.7 * continuity)
        
        # === GENERATE CELLS ===
        latent_space = []
        cluster_labels = []
        cell_types = []
        pseudotime_list = []
        
        # Blending weights
        cluster_pull_weight = (1 - continuity) ** 2
        trajectory_weight = continuity ** 0.5
        
        # For branching, split cells between two branches
        if target_trajectory == 'branching':
            n_trunk_cells = int(n_cells * branch_point)
            n_branch_cells = n_cells - n_trunk_cells
            n_branch_a_cells = n_branch_cells // 2
            n_branch_b_cells = n_branch_cells - n_branch_a_cells
            
            # === TRUNK CELLS ===
            trunk_pseudotime = np.linspace(0, branch_point, n_trunk_cells)
            for t in trunk_pseudotime:
                # Map to trunk cluster indices [0, n_trunk-1]
                cluster_pos = t / branch_point * (n_trunk - 1) if branch_point > 0 else 0
                i = min(int(cluster_pos), n_trunk - 2)
                i = max(0, i)
                j = min(i + 1, n_trunk - 1)
                local_t = cluster_pos - i
                
                # Interpolate between trunk clusters
                trajectory_point = (1 - local_t) * cluster_centers[i] + local_t * cluster_centers[j]
                nearest_cluster = i if local_t < 0.5 else j
                cluster_center = cluster_centers[nearest_cluster]
                
                # Blend
                base_position = cluster_pull_weight * cluster_center + trajectory_weight * trajectory_point
                noise_scale = cluster_noise * (1 - 0.7 * continuity)
                noise = np.random.randn(n_dims) * noise_scale
                
                latent_space.append((base_position + noise).reshape(1, -1))
                cluster_labels.append(nearest_cluster)
                cell_types.append(f'Type_{nearest_cluster}')
                pseudotime_list.append(t)
            
            # === BRANCH A CELLS ===
            branch_a_pseudotime = np.linspace(branch_point, 1.0, n_branch_a_cells, endpoint=False)
            for t in branch_a_pseudotime:
                # Map to branch A cluster indices
                t_branch = (t - branch_point) / (1 - branch_point)
                cluster_pos = t_branch * (n_branch_a - 1) if n_branch_a > 1 else 0
                i = min(int(cluster_pos), n_branch_a - 2)
                i = max(0, i)
                j = min(i + 1, n_branch_a - 1)
                local_t = cluster_pos - i
                
                idx_i = n_trunk + i
                idx_j = n_trunk + j
                
                # KEY FIX: Interpolate between last trunk cluster and first branch cluster at transition
                if t <= branch_point + 0.01 and n_trunk > 0:  # Near branch point
                    # Smooth transition from trunk to branch
                    transition_weight = (t - branch_point) / 0.01
                    trunk_point = cluster_centers[n_trunk - 1]
                    branch_point_pos = cluster_centers[idx_i]
                    trajectory_point = (1 - transition_weight) * trunk_point + transition_weight * branch_point_pos
                else:
                    trajectory_point = (1 - local_t) * cluster_centers[idx_i] + local_t * cluster_centers[idx_j]
                
                nearest_cluster = idx_i if local_t < 0.5 else idx_j
                cluster_center = cluster_centers[nearest_cluster]
                
                base_position = cluster_pull_weight * cluster_center + trajectory_weight * trajectory_point
                noise_scale = cluster_noise * (1 - 0.7 * continuity)
                noise = np.random.randn(n_dims) * noise_scale
                
                latent_space.append((base_position + noise).reshape(1, -1))
                cluster_labels.append(nearest_cluster)
                cell_types.append(f'Type_{nearest_cluster}')
                pseudotime_list.append(t)
            
            # === BRANCH B CELLS ===
            branch_b_pseudotime = np.linspace(branch_point, 1.0, n_branch_b_cells)
            for t in branch_b_pseudotime:
                # Map to branch B cluster indices
                t_branch = (t - branch_point) / (1 - branch_point)
                cluster_pos = t_branch * (n_branch_b - 1) if n_branch_b > 1 else 0
                i = min(int(cluster_pos), n_branch_b - 2)
                i = max(0, i)
                j = min(i + 1, n_branch_b - 1)
                local_t = cluster_pos - i
                
                idx_i = n_trunk + n_branch_a + i
                idx_j = n_trunk + n_branch_a + j
                
                # KEY FIX: Interpolate between last trunk cluster and first branch cluster at transition
                if t <= branch_point + 0.01 and n_trunk > 0:  # Near branch point
                    # Smooth transition from trunk to branch
                    transition_weight = (t - branch_point) / 0.01
                    trunk_point = cluster_centers[n_trunk - 1]
                    branch_point_pos = cluster_centers[idx_i]
                    trajectory_point = (1 - transition_weight) * trunk_point + transition_weight * branch_point_pos
                else:
                    trajectory_point = (1 - local_t) * cluster_centers[idx_i] + local_t * cluster_centers[idx_j]
                
                nearest_cluster = idx_i if local_t < 0.5 else idx_j
                cluster_center = cluster_centers[nearest_cluster]
                
                base_position = cluster_pull_weight * cluster_center + trajectory_weight * trajectory_point
                noise_scale = cluster_noise * (1 - 0.7 * continuity)
                noise = np.random.randn(n_dims) * noise_scale
                
                latent_space.append((base_position + noise).reshape(1, -1))
                cluster_labels.append(nearest_cluster)
                cell_types.append(f'Type_{nearest_cluster}')
                pseudotime_list.append(t)
        
        else:
            # Original logic for linear and cyclic
            pseudotime = np.linspace(0, 1, n_cells)
            
            for idx, t in enumerate(pseudotime):
                if target_trajectory == 'cyclic':
                    cluster_pos = t * n_clusters
                    i = int(cluster_pos) % n_clusters
                    j = (i + 1) % n_clusters
                    local_t = cluster_pos - int(cluster_pos)
                else:
                    cluster_pos = t * (n_clusters - 1)
                    i = min(int(cluster_pos), n_clusters - 2)
                    j = i + 1
                    local_t = cluster_pos - i
                
                trajectory_point = (1 - local_t) * cluster_centers[i] + local_t * cluster_centers[j]
                nearest_cluster = i if local_t < 0.5 else j
                cluster_center = cluster_centers[nearest_cluster]
                
                base_position = cluster_pull_weight * cluster_center + trajectory_weight * trajectory_point
                noise_scale = cluster_noise * (1 - 0.7 * continuity)
                noise = np.random.randn(n_dims) * noise_scale
                
                latent_space.append((base_position + noise).reshape(1, -1))
                cluster_labels.append(nearest_cluster)
                cell_types.append(f'Type_{nearest_cluster}')
                pseudotime_list.append(t)
        
        latent_space = np.vstack(latent_space)
        cluster_labels = np.array(cluster_labels)
        pseudotime = np.array(pseudotime_list)
        
        if return_metadata:
            return {
                'latent_space': latent_space,
                'cluster_labels': cluster_labels,
                'cell_types': cell_types,
                'pseudotime': pseudotime,
                'continuity': continuity,
                'trajectory_type': 'discrete',
                'ground_truth': {
                    'n_clusters': n_clusters,
                    'true_continuity': continuity,
                    'target_trajectory': target_trajectory,
                    'branch_point': branch_point if target_trajectory == 'branching' else None,
                    'cluster_noise': cluster_noise,
                    'separation': separation,
                    'cluster_pull_weight': cluster_pull_weight,
                    'trajectory_weight': trajectory_weight
                }
            }
        return latent_space
    
    # ==================== Noise Generation ====================
    
    def _generate_noise(self, n_samples, n_dims, noise_type, scale=1.0):
        """Generate different types of noise"""
        if noise_type == 'gaussian':
            return np.random.randn(n_samples, n_dims) * scale
        elif noise_type == 'uniform':
            return np.random.uniform(-1, 1, (n_samples, n_dims)) * scale
        elif noise_type == 'heavy_tail':
            # Student-t distribution with df=3
            return np.random.standard_t(3, (n_samples, n_dims)) * scale
        elif noise_type == 'sparse':
            noise = np.random.randn(n_samples, n_dims) * scale
            # Make 80% of entries zero
            mask = np.random.rand(n_samples, n_dims) < 0.8
            noise[mask] = 0
            return noise
        else:
            return np.random.randn(n_samples, n_dims) * scale
    
    # ==================== Helper Functions ====================
    
    def _assign_cell_types_linear(self, pseudotime):
        """Assign cell types based on pseudotime for linear trajectory"""
        types = []
        for t in pseudotime:
            if t < 0.33:
                types.append('Early')
            elif t < 0.66:
                types.append('Mid')
            else:
                types.append('Late')
        return types
    
    def _assign_cell_types_branching(self, pseudotime, branch_id):
        """Assign cell types for branching trajectory"""
        types = []
        for t, b in zip(pseudotime, branch_id):
            if b == -1:
                types.append('Progenitor')
            elif t < 0.7:
                types.append(f'Branch{b}_Early')
            else:
                types.append(f'Branch{b}_Late')
        return types
    
    def _assign_cell_types_cyclic(self, theta):
        """Assign cell cycle phases"""
        phases = []
        for t in theta:
            phase = (t % (2 * np.pi)) / (2 * np.pi)
            if phase < 0.25:
                phases.append('G1')
            elif phase < 0.5:
                phases.append('S')
            elif phase < 0.75:
                phases.append('G2')
            else:
                phases.append('M')
        return phases

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

    