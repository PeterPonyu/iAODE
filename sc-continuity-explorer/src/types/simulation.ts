/**
 * Essential Type Definitions for Single-Cell Continuity Data
 * Matches Python precomputation output structure
 */

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

/**
 * Main simulation result (matches Python output exactly)
 */
export interface SimulationResult {
  id: string;
  parameters: SimulationParameters;
  embeddings: EmbeddingData;
  metadata: SimulationMetadata;
  metrics: MetricsData;
}

export interface SimulationParameters {
  trajectory_type: TrajectoryType;
  continuity: number;
  n_cells: number;
  n_dims: number;
  replicate: number;
  global_id?: number;
}

export type TrajectoryType = 'linear' | 'branching' | 'cyclic' | 'discrete';

export type Point2D = [number, number];

export interface EmbeddingData {
  pca?: Point2D[];
  tsne?: Point2D[];
  umap?: Point2D[];
}

export interface SimulationMetadata {
  pseudotime: number[];
  cell_types: string[];
  n_cells: number;
  n_dims: number;
  trajectory_type: TrajectoryType;
  branch_id?: number[];
  cycle_phase?: number[];
  cluster_labels?: number[];
}

export interface MetricsData {
  spectral_decay: number;
  anisotropy: number;
  participation_ratio: number;
  trajectory_directionality: number;
  manifold_dimensionality: number;
  noise_resilience: number;
  variance_pca?: number;
  variance_tsne?: number;
  variance_umap?: number;
  [key: string]: number | undefined;
}

// ============================================================================
// METADATA FILES
// ============================================================================

export interface DataManifest {
  version: string;
  generation_timestamp: string;
  total_simulations: number;
  chunk_size: number;
  total_chunks: number;
  parameter_grid: {
    continuity: number[];
    trajectory_types: TrajectoryType[];
    n_cells: number[];
    n_dims: number[];
    replicates: number[];
  };
  embedding_methods: string[];
}

export interface MetricsSummary {
  [metricName: string]: {
    min: number;
    max: number;
    mean: number;
    std: number;
    median: number;
    q25: number;
    q75: number;
  };
}

export interface ParameterLookup {
  [key: string]: {
    chunk_id: number;
    index_in_chunk: number;
    simulation_id: string;
  };
}

// ============================================================================
// UI STATE
// ============================================================================

export type EmbeddingMethod = 'pca' | 'tsne' | 'umap';
export type ColorByOption = 'pseudotime' | 'cell_types' | 'branch_id' | 'cycle_phase' | 'cluster_labels';
export type ViewMode = 'single' | 'comparison' | 'grid';

export interface ExplorerState {
  continuity: number;
  trajectoryType: TrajectoryType;
  embeddingMethod: EmbeddingMethod;
  colorBy: ColorByOption;
  replicate: number;
  viewMode: ViewMode;
  showMetrics: boolean;
  isLoading: boolean;
  error: string | null;
}

// ============================================================================
// VISUALIZATION
// ============================================================================

export interface ChartConfig {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  pointSize: number;
  pointOpacity: number;
}

export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}