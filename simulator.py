import json
import gzip
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import warnings

# Import your modules
from LSE import SingleCellLatentSpaceEvaluator
from SIM import LatentSpaceSimulator

# Embedding libraries
from sklearn.decomposition import PCA
import umap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('precomputation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuityDataPrecomputer:
    """
    Comprehensive pre-computation system for single-cell continuity exploration.
    
    Generates all simulation data, embeddings, and metrics for static website consumption.
    Supports both continuous trajectories and discrete clusters with different continuity ranges.
    """
    
    def __init__(self, 
                 output_dir: str = './public/data',
                 chunk_size: int = 50,
                 compress_data: bool = True,
                 random_seed: int = 42,
                 save_ground_truth: bool = True):
        
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.compress_data = compress_data
        self.random_seed = random_seed
        self.save_ground_truth = save_ground_truth
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'chunks').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        # Initialize components
        self.simulator = LatentSpaceSimulator(random_seed=random_seed)
        self.evaluator = SingleCellLatentSpaceEvaluator()
        
        # Define metrics to compute
        self.metrics_to_compute = {
            'spectral_decay': self.evaluator.spectral_decay_rate,
            'anisotropy': self.evaluator.isotropy_anisotropy_score,
            'participation_ratio': self.evaluator.participation_ratio_score,
            'trajectory_directionality': self.evaluator.trajectory_directionality_score,
            'manifold_dimensionality': self.evaluator.manifold_dimensionality_score_v2,
            'noise_resilience': self.evaluator.noise_resilience_score
        }
        
        logger.info(f"Initialized precomputer")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Chunk size: {self.chunk_size}")
        logger.info(f"Save ground truth: {self.save_ground_truth}")
    
    def compute_single_simulation(self, params: Dict[str, Any], 
                                 embedding_methods: List[str]) -> Optional[Dict[str, Any]]:
        """
        Execute a single simulation with all computations.
        """
        try:
            # Set random seed for reproducibility
            sim_seed = self.random_seed + params.get('global_id', 0)
            np.random.seed(sim_seed)
            
            # Generate simulation data
            sim_result = self._generate_simulation_data(params)
            if sim_result is None:
                return None
            
            latent_space = sim_result['latent_space']
            
            # Compute embeddings
            embeddings = self._compute_embeddings(latent_space, sim_seed, embedding_methods)
            
            # Compute metrics
            metrics = self._compute_metrics(latent_space)
            
            # Additional derived metrics from embeddings
            embedding_metrics = self._compute_embedding_metrics(embeddings)
            
            # Build comprehensive metadata
            metadata = self._build_metadata(sim_result, params)
            
            # Package complete result
            result = {
                'id': self._generate_simulation_id(params),
                'parameters': params.copy(),
                'embeddings': embeddings,
                'metadata': metadata,
                'metrics': {**metrics, **embedding_metrics}
            }
            
            # Optionally include ground truth information
            if self.save_ground_truth and 'ground_truth' in sim_result:
                result['ground_truth'] = sim_result['ground_truth']
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute simulation {params}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_simulation_data(self, params: Dict[str, Any]) -> Optional[Dict]:
        """Generate the actual simulation data based on trajectory type"""
        traj_type = params['trajectory_type']
        
        try:
            if traj_type == 'linear':
                return self.simulator.simulate_linear_trajectory(
                    n_cells=params['n_cells'],
                    n_dims=params['n_dims'],
                    continuity=params['continuity'],
                    return_metadata=True
                )
            
            elif traj_type == 'branching':
                return self.simulator.simulate_branching_trajectory(
                    n_cells=params['n_cells'],
                    n_dims=params['n_dims'],
                    continuity=params['continuity'],
                    branch_point=params.get('branch_point', 0.4),
                    n_branches=params.get('n_branches', 2),
                    branch_angle=params.get('branch_angle', 60),
                    return_metadata=True
                )
            
            elif traj_type == 'cyclic':
                return self.simulator.simulate_cyclic_trajectory(
                    n_cells=params['n_cells'],
                    n_dims=params['n_dims'],
                    continuity=params['continuity'],
                    n_cycles=params.get('n_cycles', 1.5),
                    return_metadata=True
                )
            
            elif traj_type == 'discrete':
                # Discrete clusters with underlying trajectory structure
                return self.simulator.simulate_discrete_clusters(
                    n_cells=params['n_cells'],
                    n_clusters=params.get('n_clusters', 5),
                    n_dims=params['n_dims'],
                    continuity=params['continuity'],
                    base_separation=params.get('base_separation', 5.0),
                    target_trajectory=params.get('target_trajectory', 'linear'),
                    branch_point=params.get('branch_point', 0.5),
                    return_metadata=True
                )
            
            else:
                logger.error(f"Unknown trajectory type: {traj_type}")
                return None
                
        except Exception as e:
            logger.error(f"Simulation generation failed for {params}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _build_metadata(self, sim_result: Dict, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive metadata structure for all trajectory types"""
        
        # Base metadata common to all types
        metadata = {
            'pseudotime': self._safe_array_to_list(sim_result.get('pseudotime')),
            'cell_types': sim_result.get('cell_types', []),
            'n_cells': params['n_cells'],
            'n_dims': params['n_dims'],
            'trajectory_type': params['trajectory_type'],
            'continuity': params['continuity']
        }
        
        # Add trajectory-specific fields
        traj_type = params['trajectory_type']
        
        if traj_type == 'branching':
            if 'branch_id' in sim_result:
                metadata['branch_id'] = self._safe_array_to_list(sim_result['branch_id'])
            metadata['n_branches'] = params.get('n_branches', 2)
            metadata['branch_point'] = params.get('branch_point', 0.4)
            metadata['branch_angle'] = params.get('branch_angle', 60)
                
        elif traj_type == 'cyclic':
            if 'cycle_phase' in sim_result:
                metadata['cycle_phase'] = self._safe_array_to_list(sim_result['cycle_phase'])
            metadata['n_cycles'] = params.get('n_cycles', 1.5)
                
        elif traj_type == 'discrete':
            if 'cluster_labels' in sim_result:
                metadata['cluster_labels'] = self._safe_array_to_list(sim_result['cluster_labels'])
            metadata['n_clusters'] = params.get('n_clusters', 5)
            metadata['target_trajectory'] = params.get('target_trajectory', 'linear')
            metadata['base_separation'] = params.get('base_separation', 5.0)
            
            # Add branching info if discrete clusters follow branching pattern
            if params.get('target_trajectory') == 'branching':
                metadata['branch_point'] = params.get('branch_point', 0.5)
        
        return metadata
    
    def _compute_embeddings(self, latent_space: np.ndarray, seed: int,
                          methods: List[str]) -> Dict[str, List[List[float]]]:
        """Compute specified embedding methods (PCA and UMAP only)"""
        embeddings = {}
        
        for method in methods:
            method_lower = method.lower()
            
            try:
                if method_lower == 'pca':
                    pca = PCA(n_components=2, random_state=seed)
                    embed = pca.fit_transform(latent_space)
                    embeddings['pca'] = embed.tolist()
                    
                elif method_lower == 'umap':
                    n_neighbors = min(15, max(2, latent_space.shape[0] // 3))
                    umap_reducer = umap.UMAP(
                        n_components=2, 
                        random_state=seed,
                        n_neighbors=n_neighbors,
                        min_dist=0.1
                    )
                    embed = umap_reducer.fit_transform(latent_space)
                    embeddings['umap'] = embed.tolist()
                    
                else:
                    logger.warning(f"Unknown or removed embedding method: {method}")
                    
            except Exception as e:
                logger.warning(f"{method} embedding computation failed: {str(e)}")
                # Return empty embedding if computation fails
                n_cells = latent_space.shape[0]
                embeddings[method_lower] = [[0.0, 0.0]] * n_cells
        
        return embeddings
    
    def _compute_metrics(self, latent_space: np.ndarray) -> Dict[str, float]:
        """Compute all continuity metrics"""
        metrics = {}
        
        for metric_name, metric_func in self.metrics_to_compute.items():
            try:
                score = metric_func(latent_space)
                # Handle potential NaN or infinite values
                if np.isfinite(score):
                    metrics[metric_name] = float(score)
                else:
                    logger.warning(f"Invalid score for {metric_name}: {score}")
                    metrics[metric_name] = 0.0
            except Exception as e:
                logger.warning(f"Metric computation failed for {metric_name}: {str(e)}")
                metrics[metric_name] = 0.0
        
        return metrics
    
    def _compute_embedding_metrics(self, embeddings: Dict) -> Dict[str, float]:
        """Compute additional metrics on embeddings"""
        embedding_metrics = {}
        
        try:
            # Compute embedding spread/variance
            for embed_method, embed_data in embeddings.items():
                try:
                    embed_array = np.array(embed_data)
                    variance = np.var(embed_array, axis=0).mean()
                    embedding_metrics[f'variance_{embed_method}'] = float(variance)
                except:
                    embedding_metrics[f'variance_{embed_method}'] = 0.0
                    
        except Exception as e:
            logger.warning(f"Embedding metrics computation failed: {str(e)}")
        
        return embedding_metrics
    
    def _generate_simulation_id(self, params: Dict[str, Any]) -> str:
        """Generate unique ID for simulation"""
        traj_type = params['trajectory_type']
        cont = params['continuity']
        n_cells = params['n_cells']
        rep = params['replicate']
        
        base_id = f"{traj_type}_cont{cont:.3f}_n{n_cells}_rep{rep}"
        
        # Add type-specific identifiers
        if traj_type == 'discrete':
            target_traj = params.get('target_trajectory', 'linear')
            n_clusters = params.get('n_clusters', 5)
            base_id += f"_target{target_traj}_k{n_clusters}"
        elif traj_type == 'branching':
            n_branches = params.get('n_branches', 2)
            base_id += f"_br{n_branches}"
        elif traj_type == 'cyclic':
            n_cycles = params.get('n_cycles', 1.5)
            base_id += f"_cyc{n_cycles:.1f}"
        
        return base_id
    
    def _safe_array_to_list(self, arr) -> List:
        """Safely convert numpy array to list"""
        if arr is None:
            return []
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        if isinstance(arr, (list, tuple)):
            return list(arr)
        return [arr]
    
    def run_precomputation(self,
                          continuous_config: Optional[Dict[str, Any]] = None,
                          discrete_config: Optional[Dict[str, Any]] = None,
                          n_cells: List[int] = [500],
                          n_dims: int = 50,
                          embedding_methods: List[str] = ['pca', 'umap']) -> Dict[str, Any]:
        """
        Execute complete precomputation pipeline with separate configs for continuous and discrete.
        
        Parameters:
            continuous_config: Configuration for continuous trajectories
                {
                    'trajectory_types': ['linear', 'branching', 'cyclic'],
                    'continuity_levels': [0.85, 0.90, ..., 0.99],
                    'n_replicates': 1,
                    'branching_params': {'n_branches': 3, 'branch_point': 0.5, 'branch_angle': 60},
                    'cyclic_params': {'n_cycles': 2}
                }
            discrete_config: Configuration for discrete clusters
                {
                    'continuity_levels': [0.0, 0.1, ..., 1.0],
                    'n_clusters_list': [8, 10, 12],
                    'target_trajectories': ['linear', 'branching', 'cyclic'],
                    'n_replicates': 1,
                    'base_separation': 5.0,
                    'branch_point': 0.5
                }
            n_cells: List of cell counts
            n_dims: Dimensionality of latent space
            embedding_methods: List of embedding methods to compute
        """
        start_time = datetime.now()
        
        # Validate embedding methods
        valid_methods = ['pca', 'umap']
        embedding_methods = [m.lower() for m in embedding_methods if m.lower() in valid_methods]
        if not embedding_methods:
            raise ValueError(f"No valid embedding methods specified. Valid options: {valid_methods}")
        
        # Build parameter combinations
        all_param_combinations = []
        
        # Add continuous trajectory simulations
        if continuous_config is not None:
            continuous_params = self._expand_continuous_parameters(
                continuous_config, n_cells, n_dims
            )
            all_param_combinations.extend(continuous_params)
            logger.info(f"Generated {len(continuous_params)} continuous trajectory simulations")
        
        # Add discrete trajectory simulations
        if discrete_config is not None:
            discrete_params = self._expand_discrete_parameters(
                discrete_config, n_cells, n_dims
            )
            all_param_combinations.extend(discrete_params)
            logger.info(f"Generated {len(discrete_params)} discrete cluster simulations")
        
        if not all_param_combinations:
            raise ValueError("No simulations configured. Provide continuous_config and/or discrete_config.")
        
        total_simulations = len(all_param_combinations)
        logger.info(f"Starting precomputation of {total_simulations} total simulations")
        logger.info(f"Cell counts: {n_cells}")
        logger.info(f"Embedding methods: {embedding_methods}")
        
        # Add global ID for reproducible seeding
        for i, params in enumerate(all_param_combinations):
            params['global_id'] = i
        
        # Execute simulations sequentially
        results = []
        for i, params in enumerate(all_param_combinations):
            result = self.compute_single_simulation(params, embedding_methods)
            results.append(result)
            
            if (i + 1) % 10 == 0 or (i + 1) == total_simulations:
                logger.info(f"Completed {i + 1}/{total_simulations} simulations ({100*(i+1)/total_simulations:.1f}%)")
        
        # Filter successful results
        successful_results = [r for r in results if r is not None]
        failed_count = len(results) - len(successful_results)
        
        if failed_count > 0:
            logger.warning(f"{failed_count} simulations failed")
        
        # Save results in chunks
        self._save_results_in_chunks(successful_results)
        
        # Generate metadata and manifests
        computation_summary = self._generate_metadata(
            continuous_config, discrete_config, successful_results, 
            start_time, embedding_methods
        )
        
        logger.info(f"Precomputation completed in {datetime.now() - start_time}")
        logger.info(f"Successfully computed {len(successful_results)}/{total_simulations} simulations")
        
        return computation_summary
    
    def _expand_continuous_parameters(self, 
                                     config: Dict[str, Any],
                                     n_cells: List[int],
                                     n_dims: int) -> List[Dict[str, Any]]:
        """Expand parameter grid for continuous trajectories"""
        combinations = []
        
        trajectory_types = config.get('trajectory_types', ['linear', 'branching', 'cyclic'])
        continuity_levels = config.get('continuity_levels', [0.9])
        n_replicates = config.get('n_replicates', 1)
        branching_params = config.get('branching_params', {
            'n_branches': 2, 
            'branch_point': 0.4, 
            'branch_angle': 60
        })
        cyclic_params = config.get('cyclic_params', {'n_cycles': 1.5})
        
        for traj_type in trajectory_types:
            for continuity in continuity_levels:
                for n_cell in n_cells:
                    for replicate in range(n_replicates):
                        
                        params = {
                            'trajectory_type': traj_type,
                            'continuity': continuity,
                            'n_cells': n_cell,
                            'n_dims': n_dims,
                            'replicate': replicate
                        }
                        
                        # Add trajectory-specific parameters
                        if traj_type == 'branching':
                            params.update(branching_params)
                        elif traj_type == 'cyclic':
                            params.update(cyclic_params)
                        
                        combinations.append(params)
        
        return combinations
    
    def _expand_discrete_parameters(self,
                                   config: Dict[str, Any],
                                   n_cells: List[int],
                                   n_dims: int) -> List[Dict[str, Any]]:
        """Expand parameter grid for discrete cluster simulations"""
        combinations = []
        
        continuity_levels = config.get('continuity_levels', [0.0, 0.5, 1.0])
        n_clusters_list = config.get('n_clusters_list', [5])
        target_trajectories = config.get('target_trajectories', ['linear'])
        n_replicates = config.get('n_replicates', 1)
        base_separation = config.get('base_separation', 5.0)
        branch_point = config.get('branch_point', 0.5)
        
        for continuity in continuity_levels:
            for n_clusters in n_clusters_list:
                for target_traj in target_trajectories:
                    for n_cell in n_cells:
                        for replicate in range(n_replicates):
                            
                            params = {
                                'trajectory_type': 'discrete',
                                'continuity': continuity,
                                'n_cells': n_cell,
                                'n_dims': n_dims,
                                'replicate': replicate,
                                'n_clusters': n_clusters,
                                'target_trajectory': target_traj,
                                'base_separation': base_separation
                            }
                            
                            # Add branch_point for branching target trajectories
                            if target_traj == 'branching':
                                params['branch_point'] = branch_point
                            
                            combinations.append(params)
        
        return combinations
    
    def _save_results_in_chunks(self, results: List[Dict]):
        """Save results in chunks for efficient loading"""
        total_chunks = (len(results) + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(results))
            chunk_data = results[start_idx:end_idx]
            
            # Save uncompressed version
            chunk_path = self.output_dir / 'chunks' / f'chunk_{chunk_idx}.json'
            with open(chunk_path, 'w') as f:
                json.dump(chunk_data, f, indent=2)
            
            # Save compressed version
            if self.compress_data:
                with gzip.open(str(chunk_path) + '.gz', 'wt') as f:
                    json.dump(chunk_data, f)
            
            logger.info(f"Saved chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_data)} simulations)")
    
    def _generate_metadata(self, 
                          continuous_config: Optional[Dict],
                          discrete_config: Optional[Dict],
                          results: List[Dict], 
                          start_time: datetime, 
                          embedding_methods: List[str]) -> Dict[str, Any]:
        """Generate comprehensive metadata files"""
        
        # Main manifest
        manifest = {
            'version': '1.0',
            'generation_timestamp': datetime.now().isoformat(),
            'computation_duration_seconds': (datetime.now() - start_time).total_seconds(),
            'total_simulations': len(results),
            'chunk_size': self.chunk_size,
            'total_chunks': (len(results) + self.chunk_size - 1) // self.chunk_size,
            'random_seed': self.random_seed,
            'metrics_computed': list(self.metrics_to_compute.keys()),
            'embedding_methods': embedding_methods,
            'save_ground_truth': self.save_ground_truth,
            'continuous_config': continuous_config,
            'discrete_config': discrete_config
        }
        
        # Save main manifest
        with open(self.output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Generate parameter lookup table for fast searches
        param_lookup = {}
        for i, result in enumerate(results):
            simulation_id = result['id']
            param_lookup[simulation_id] = {
                'chunk_id': i // self.chunk_size,
                'index_in_chunk': i % self.chunk_size,
                'parameters': result['parameters']
            }
        
        with open(self.output_dir / 'metadata' / 'parameter_lookup.json', 'w') as f:
            json.dump(param_lookup, f, indent=2)
        
        # Generate metrics summary
        metrics_summary = self._compute_metrics_summary(results)
        with open(self.output_dir / 'metadata' / 'metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Generate continuity level index
        continuity_index = self._generate_continuity_index(results)
        with open(self.output_dir / 'metadata' / 'continuity_index.json', 'w') as f:
            json.dump(continuity_index, f, indent=2)
        
        # Generate trajectory type index
        trajectory_index = self._generate_trajectory_index(results)
        with open(self.output_dir / 'metadata' / 'trajectory_index.json', 'w') as f:
            json.dump(trajectory_index, f, indent=2)
        
        logger.info("Generated all metadata files")
        
        return {
            'manifest': manifest,
            'metrics_summary': metrics_summary,
            'continuity_index': continuity_index,
            'trajectory_index': trajectory_index
        }
    
    def _compute_metrics_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics for all metrics"""
        all_metrics = {}
        
        # Collect all metric values
        for result in results:
            for metric_name, value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Compute statistics
        summary = {}
        for metric_name, values in all_metrics.items():
            values = np.array(values)
            summary[metric_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        
        return summary
    
    def _generate_continuity_index(self, results: List[Dict]) -> Dict[str, List]:
        """Generate index for fast continuity-based lookups"""
        continuity_index = {}
        
        for result in results:
            params = result['parameters']
            cont_level = f"{params['continuity']:.3f}"
            traj_type = params['trajectory_type']
            
            key = f"{traj_type}_{cont_level}"
            if key not in continuity_index:
                continuity_index[key] = []
            
            continuity_index[key].append({
                'simulation_id': result['id'],
                'replicate': params['replicate'],
                'n_cells': params['n_cells']
            })
        
        return continuity_index
    
    def _generate_trajectory_index(self, results: List[Dict]) -> Dict[str, List]:
        """Generate index organized by trajectory type"""
        trajectory_index = {}
        
        for result in results:
            traj_type = result['parameters']['trajectory_type']
            
            if traj_type not in trajectory_index:
                trajectory_index[traj_type] = []
            
            entry = {
                'simulation_id': result['id'],
                'continuity': result['parameters']['continuity'],
                'n_cells': result['parameters']['n_cells'],
                'replicate': result['parameters']['replicate']
            }
            
            # Add type-specific information
            if traj_type == 'discrete':
                entry['n_clusters'] = result['parameters'].get('n_clusters')
                entry['target_trajectory'] = result['parameters'].get('target_trajectory')
            
            trajectory_index[traj_type].append(entry)
        
        return trajectory_index


def main():
    """Main execution function with separate configs for continuous and discrete trajectories"""
    
    # ==================== USER CONFIGURATION ====================
    
    # Output directory
    OUTPUT_DIR = './public/data'
    
    # Cell counts to simulate
    N_CELLS = [500]
    
    # Latent space dimensionality
    N_DIMS = 50
    
    # Embedding methods to compute (t-SNE removed for performance)
    EMBEDDING_METHODS = ['pca', 'umap']
    
    # Chunk size for file storage
    CHUNK_SIZE = 50
    
    # Save ground truth metadata (useful for validation)
    SAVE_GROUND_TRUTH = True
    
    # ========== CONTINUOUS TRAJECTORY CONFIGURATION ==========
    CONTINUOUS_CONFIG = {
        'trajectory_types': ['linear', 'branching', 'cyclic'],
        'continuity_levels': np.linspace(0.85, 0.99, 20).tolist(),  # 15 levels from 0.85 to 0.99
        'n_replicates': 1,
        'branching_params': {
            'n_branches': [2, 3, 4],
            'branch_point': 0.5,
            'branch_angle': 60
        },
        'cyclic_params': {
            'n_cycles': [2, 3, 4] 
        }
    }
    
    # ========== DISCRETE CLUSTER CONFIGURATION ==========
    DISCRETE_CONFIG = {
        'continuity_levels': np.linspace(0.0, 1.0, 20).tolist(),  # 15 levels from 0.0 to 1.0
        'n_clusters_list': [8, 10, 12],
        'target_trajectories': ['linear', 'branching', 'cyclic'],  # All three underlying structures
        'n_replicates': 1,
        'base_separation': 5.0,
        'branch_point': 0.5  # For branching target trajectories
    }
    
    # ============================================================
    
    print("=" * 80)
    print("SINGLE-CELL CONTINUITY DATA PRE-COMPUTATION")
    print("=" * 80)
    
    # Calculate totals
    n_continuous = 0
    if CONTINUOUS_CONFIG:
        n_traj_types = len(CONTINUOUS_CONFIG['trajectory_types'])
        n_cont_levels = len(CONTINUOUS_CONFIG['continuity_levels'])
        n_reps = CONTINUOUS_CONFIG['n_replicates']
        n_continuous = n_traj_types * n_cont_levels * len(N_CELLS) * n_reps
        
        print("\n--- CONTINUOUS TRAJECTORIES ---")
        print(f"Types: {CONTINUOUS_CONFIG['trajectory_types']}")
        print(f"Continuity range: {min(CONTINUOUS_CONFIG['continuity_levels']):.2f} - {max(CONTINUOUS_CONFIG['continuity_levels']):.2f} ({n_cont_levels} levels)")
        print(f"Replicates: {n_reps}")
        print(f"Branching params: {CONTINUOUS_CONFIG['branching_params']}")
        print(f"Cyclic params: {CONTINUOUS_CONFIG['cyclic_params']}")
        print(f"Total continuous simulations: {n_continuous}")
    
    n_discrete = 0
    if DISCRETE_CONFIG:
        n_cont_levels_disc = len(DISCRETE_CONFIG['continuity_levels'])
        n_clusters_variants = len(DISCRETE_CONFIG['n_clusters_list'])
        n_target_traj = len(DISCRETE_CONFIG['target_trajectories'])
        n_reps_disc = DISCRETE_CONFIG['n_replicates']
        n_discrete = n_cont_levels_disc * n_clusters_variants * n_target_traj * len(N_CELLS) * n_reps_disc
        
        print("\n--- DISCRETE CLUSTERS ---")
        print(f"Continuity range: {min(DISCRETE_CONFIG['continuity_levels']):.2f} - {max(DISCRETE_CONFIG['continuity_levels']):.2f} ({n_cont_levels_disc} levels)")
        print(f"N clusters: {DISCRETE_CONFIG['n_clusters_list']}")
        print(f"Target trajectories: {DISCRETE_CONFIG['target_trajectories']}")
        print(f"Replicates: {n_reps_disc}")
        print(f"Base separation: {DISCRETE_CONFIG['base_separation']}")
        print(f"Total discrete simulations: {n_discrete}")
    
    total_simulations = n_continuous + n_discrete
    print(f"\n{'='*80}")
    print(f"TOTAL SIMULATIONS: {total_simulations}")
    print(f"  - Continuous: {n_continuous}")
    print(f"  - Discrete: {n_discrete}")
    print(f"Cell counts: {N_CELLS}")
    print(f"Embedding methods: {EMBEDDING_METHODS}")
    print("=" * 80)
    
    # Confirmation prompt
    response = input("\nProceed with precomputation? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Initialize precomputer
    precomputer = ContinuityDataPrecomputer(
        output_dir=OUTPUT_DIR,
        chunk_size=CHUNK_SIZE,
        compress_data=True,
        random_seed=42,
        save_ground_truth=SAVE_GROUND_TRUTH
    )
    
    # Run precomputation
    try:
        summary = precomputer.run_precomputation(
            continuous_config=CONTINUOUS_CONFIG,
            discrete_config=DISCRETE_CONFIG,
            n_cells=N_CELLS,
            n_dims=N_DIMS,
            embedding_methods=EMBEDDING_METHODS
        )
        
        print("\n" + "=" * 80)
        print("PRECOMPUTATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total simulations: {summary['manifest']['total_simulations']}")
        print(f"Data directory: {OUTPUT_DIR}")
        print(f"Chunks generated: {summary['manifest']['total_chunks']}")
        print(f"Duration: {summary['manifest']['computation_duration_seconds']:.1f} seconds")
        print(f"Average time per simulation: {summary['manifest']['computation_duration_seconds']/summary['manifest']['total_simulations']:.2f} seconds")
        print("\nGenerated metadata files:")
        print("  - manifest.json")
        print("  - metadata/parameter_lookup.json")
        print("  - metadata/metrics_summary.json")
        print("  - metadata/continuity_index.json")
        print("  - metadata/trajectory_index.json")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Precomputation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()