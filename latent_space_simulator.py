import numpy as np
from typing import Dict, List, Union


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
                                   return_metadata: bool = True) -> Union[Dict, np.ndarray]:
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
                                     return_metadata: bool = True) -> Union[Dict, np.ndarray]:
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
                                  return_metadata: bool = True) -> Union[Dict, np.ndarray]:
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
                              return_metadata: bool = True) -> Union[Dict, np.ndarray]:
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
