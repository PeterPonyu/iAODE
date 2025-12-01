// ============================================================================
// lib/dataLoader.ts - Data Loading from Precomputed Simulations
// ============================================================================

import { 
  SimulationResult, 
  TrajectoryType, 
  DataManifest,
  ParameterLookup,
  ContinuityIndex,
  MetricsSummary
} from '@/types/simulation';

// ============================================
// CONFIGURATION
// ============================================

// When basePath is configured in next.config.ts, static files in public/
// are also served under that basePath in BOTH development and production.
// So we always use the full path with basePath.
const DATA_BASE_PATH = '/iAODE/explorer/data';
const USE_COMPRESSION = false; // Set to true if using .gz files

// ============================================
// CACHED DATA
// ============================================

let manifestCache: DataManifest | null = null;
let parameterLookupCache: ParameterLookup | null = null;
let continuityIndexCache: ContinuityIndex | null = null;
let metricsSummaryCache: MetricsSummary | null = null;

// Chunk cache: Map<chunk_id, SimulationResult[]>
const chunkCache = new Map<number, SimulationResult[]>();

// ============================================
// METADATA LOADING
// ============================================

/**
 * Load the main manifest file
 */
async function loadManifest(): Promise<DataManifest> {
  if (manifestCache) return manifestCache;
  
  try {
    const response = await fetch(`${DATA_BASE_PATH}/manifest.json`);
    if (!response.ok) {
      throw new Error(`Failed to load manifest: ${response.statusText}`);
    }
    const data = await response.json();
    manifestCache = data;
    return data;
  } catch (error) {
    console.error('Error loading manifest:', error);
    throw new Error('Failed to load data manifest. Ensure data has been precomputed.');
  }
}

/**
 * Load the parameter lookup table
 */
async function loadParameterLookup(): Promise<ParameterLookup> {
  if (parameterLookupCache) return parameterLookupCache;
  
  try {
    const response = await fetch(`${DATA_BASE_PATH}/metadata/parameter_lookup.json`);
    if (!response.ok) {
      throw new Error(`Failed to load parameter lookup: ${response.statusText}`);
    }
    const data = await response.json();
    parameterLookupCache = data;
    return data;
  } catch (error) {
    console.error('Error loading parameter lookup:', error);
    throw new Error('Failed to load parameter lookup');
  }
}

/**
 * Load the continuity index
 */
async function loadContinuityIndex(): Promise<ContinuityIndex> {
  if (continuityIndexCache) return continuityIndexCache;
  
  try {
    const response = await fetch(`${DATA_BASE_PATH}/metadata/continuity_index.json`);
    if (!response.ok) {
      throw new Error(`Failed to load continuity index: ${response.statusText}`);
    }
    const data = await response.json();
    continuityIndexCache = data;
    return data;
  } catch (error) {
    console.error('Error loading continuity index:', error);
    throw new Error('Failed to load continuity index');
  }
}

/**
 * Load metrics summary
 */
export async function loadMetricsSummary(): Promise<MetricsSummary> {
  if (metricsSummaryCache) return metricsSummaryCache;
  
  try {
    const response = await fetch(`${DATA_BASE_PATH}/metadata/metrics_summary.json`);
    if (!response.ok) {
      throw new Error(`Failed to load metrics summary: ${response.statusText}`);
    }
    const data = await response.json();
    metricsSummaryCache = data;
    return data;
  } catch (error) {
    console.error('Error loading metrics summary:', error);
    throw new Error('Failed to load metrics summary');
  }
}

// ============================================
// CHUNK LOADING
// ============================================

/**
 * Load a specific data chunk
 */
async function loadChunk(chunkId: number): Promise<SimulationResult[]> {
  // Check cache first
  if (chunkCache.has(chunkId)) {
    return chunkCache.get(chunkId)!;
  }
  
  try {
    const extension = USE_COMPRESSION ? '.json.gz' : '.json';
    const response = await fetch(`${DATA_BASE_PATH}/chunks/chunk_${chunkId}${extension}`);
    
    if (!response.ok) {
      throw new Error(`Failed to load chunk ${chunkId}: ${response.statusText}`);
    }
    
    const data: SimulationResult[] = await response.json();
    
    // Cache the chunk
    chunkCache.set(chunkId, data);
    
    return data;
  } catch (error) {
    console.error(`Error loading chunk ${chunkId}:`, error);
    throw new Error(`Failed to load data chunk ${chunkId}`);
  }
}

// ============================================
// MAIN LOADING FUNCTION
// ============================================

/**
 * Load a specific simulation by parameters
 * 
 * @param trajectoryType - Type of trajectory
 * @param continuity - Continuity value (0.0 - 1.0)
 * @param replicate - Replicate number (0, 1, 2, ...)
 * @param nCells - Number of cells (optional, defaults to 500)
 * @param n_branches - Number of branches (for branching trajectories)
 * @param n_cycles - Number of cycles (for cyclic trajectories)
 * @param n_clusters - Number of clusters (for discrete trajectories)
 * @param target_trajectory - Target trajectory type (for discrete trajectories)
 * @returns Promise<SimulationResult>
 */
export async function loadSimulation(
  trajectoryType: TrajectoryType,
  continuity: number,
  replicate: number,
  nCells: number = 500,
  n_branches?: number,
  n_cycles?: number,
  n_clusters?: number,
  target_trajectory?: TrajectoryType
): Promise<SimulationResult> {
  try {
    // Load parameter lookup
    const paramLookup = await loadParameterLookup();
    
    // Construct simulation ID (must match Python's _generate_simulation_id)
    const simulationId = constructSimulationId(
      trajectoryType, 
      continuity, 
      nCells, 
      replicate,
      n_branches,
      n_cycles,
      n_clusters,
      target_trajectory
    );
    
    // Find the simulation in the lookup
    const lookupEntry = paramLookup[simulationId];
    
    if (!lookupEntry) {
      // Try to find a close match
      const closestId = findClosestSimulation(
        paramLookup,
        trajectoryType,
        continuity,
        nCells,
        replicate,
        n_branches,
        n_cycles,
        n_clusters,
        target_trajectory
      );
      
      if (closestId) {
        console.warn(`Exact simulation not found. Using closest match: ${closestId}`);
        return loadSimulationById(closestId);
      }
      
      throw new Error(
        `Simulation not found: ${simulationId}`
      );
    }
    
    // Load the chunk containing this simulation
    const chunk = await loadChunk(lookupEntry.chunk_id);
    
    // Get the specific simulation from the chunk
    const simulation = chunk[lookupEntry.index_in_chunk];
    
    if (!simulation) {
      throw new Error(`Simulation not found in chunk at index ${lookupEntry.index_in_chunk}`);
    }
    
    return simulation;
    
  } catch (error) {
    console.error('Error loading simulation:', error);
    throw error;
  }
}

/**
 * Load a simulation by its ID
 */
export async function loadSimulationById(simulationId: string): Promise<SimulationResult> {
  try {
    const paramLookup = await loadParameterLookup();
    const lookupEntry = paramLookup[simulationId];
    
    if (!lookupEntry) {
      throw new Error(`Simulation ID not found: ${simulationId}`);
    }
    
    const chunk = await loadChunk(lookupEntry.chunk_id);
    const simulation = chunk[lookupEntry.index_in_chunk];
    
    if (!simulation) {
      throw new Error(`Simulation not found in chunk`);
    }
    
    return simulation;
    
  } catch (error) {
    console.error('Error loading simulation by ID:', error);
    throw error;
  }
}

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Construct simulation ID matching Python's format
 */
function constructSimulationId(
  trajectoryType: TrajectoryType,
  continuity: number,
  nCells: number,
  replicate: number,
  n_branches?: number,
  n_cycles?: number,
  n_clusters?: number,
  target_trajectory?: TrajectoryType
): string {
  let id = `${trajectoryType}_cont${continuity.toFixed(3)}_n${nCells}_rep${replicate}`;
  
  // Add trajectory-specific suffixes (matching Python)
  if (trajectoryType === 'discrete') {
    const targetTraj = target_trajectory || 'linear';
    const clusters = n_clusters || 5;
    id += `_target${targetTraj}_k${clusters}`;
  } else if (trajectoryType === 'branching') {
    const branches = n_branches || 2;
    id += `_br${branches}`;
  } else if (trajectoryType === 'cyclic') {
    const cycles = n_cycles || 1.5;
    id += `_cyc${cycles.toFixed(1)}`;
  }
  
  return id;
}

/**
 * Find the closest matching simulation when exact match not found
 */
function findClosestSimulation(
  paramLookup: ParameterLookup,
  trajectoryType: TrajectoryType,
  targetContinuity: number,
  nCells: number,
  replicate: number,
  n_branches?: number,
  n_cycles?: number,
  n_clusters?: number,
  target_trajectory?: TrajectoryType
): string | null {
  let closestId: string | null = null;
  let minDiff = Infinity;
  
  for (const [id, entry] of Object.entries(paramLookup)) {
    const params = entry.parameters;
    
    // Match trajectory type, n_cells, and replicate exactly
    if (
      params.trajectory_type === trajectoryType &&
      params.n_cells === nCells &&
      params.replicate === replicate
    ) {
      // Match trajectory-specific parameters
      let paramsMatch = true;
      
      if (trajectoryType === 'branching' && n_branches !== undefined) {
        paramsMatch = params.n_branches === n_branches;
      } else if (trajectoryType === 'cyclic' && n_cycles !== undefined) {
        paramsMatch = params.n_cycles === n_cycles;
      } else if (trajectoryType === 'discrete') {
        if (n_clusters !== undefined) {
          paramsMatch = paramsMatch && params.n_clusters === n_clusters;
        }
        if (target_trajectory !== undefined) {
          paramsMatch = paramsMatch && params.target_trajectory === target_trajectory;
        }
      }
      
      if (paramsMatch) {
        const diff = Math.abs(params.continuity - targetContinuity);
        if (diff < minDiff) {
          minDiff = diff;
          closestId = id;
        }
      }
    }
  }
  
  return closestId;
}

// ============================================
// AVAILABLE OPTIONS QUERIES
// ============================================

/**
 * Get all available continuity values from the manifest
 */
export async function getAvailableContinuityValues(): Promise<number[]> {
  try {
    const manifest = await loadManifest();
    
    // Combine continuous and discrete continuity levels
    const continuitySet = new Set<number>();
    
    if (manifest.continuous_config?.continuity_levels) {
      manifest.continuous_config.continuity_levels.forEach((c: number) => continuitySet.add(c));
    }
    
    if (manifest.discrete_config?.continuity_levels) {
      manifest.discrete_config.continuity_levels.forEach((c: number) => continuitySet.add(c));
    }
    
    // Convert to sorted array
    const continuities = Array.from(continuitySet).sort((a, b) => a - b);
    
    return continuities.length > 0 ? continuities : [0.9, 0.95, 0.99];
    
  } catch (error) {
    console.error('Error getting available continuity values:', error);
    // Return default values as fallback
    return [0.85, 0.90, 0.95, 0.99];
  }
}

/**
 * Get all available trajectory types
 */
export async function getAvailableTrajectoryTypes(): Promise<TrajectoryType[]> {
  try {
    const manifest = await loadManifest();
    
    const trajectoryTypes = new Set<TrajectoryType>();
    
    // Add continuous trajectory types
    if (manifest.continuous_config?.trajectory_types) {
      manifest.continuous_config.trajectory_types.forEach((t: string) => 
        trajectoryTypes.add(t as TrajectoryType)
      );
    }
    
    // Discrete trajectories always have type 'discrete'
    if (manifest.discrete_config) {
      trajectoryTypes.add('discrete');
    }
    
    return Array.from(trajectoryTypes);
    
  } catch (error) {
    console.error('Error getting available trajectory types:', error);
    return ['linear', 'branching', 'cyclic'];
  }
}

/**
 * Get available replicates
 */
export async function getAvailableReplicates(): Promise<number[]> {
  try {
    const manifest = await loadManifest();
    
    // Get max replicates from configs
    const continuousReps = manifest.continuous_config?.n_replicates || 1;
    const discreteReps = manifest.discrete_config?.n_replicates || 1;
    
    const maxReps = Math.max(continuousReps, discreteReps);
    
    // Return array [0, 1, 2, ..., maxReps-1]
    return Array.from({ length: maxReps }, (_, i) => i);
    
  } catch (error) {
    console.error('Error getting available replicates:', error);
    return [0, 1, 2];
  }
}

/**
 * Get available cell counts
 */
export async function getAvailableCellCounts(): Promise<number[]> {
  try {
    const paramLookup = await loadParameterLookup();
    
    const cellCounts = new Set<number>();
    
    for (const entry of Object.values(paramLookup)) {
      cellCounts.add(entry.parameters.n_cells);
    }
    
    return Array.from(cellCounts).sort((a, b) => a - b);
    
  } catch (error) {
    console.error('Error getting available cell counts:', error);
    return [500];
  }
}

/**
 * Get available n_branches values for branching trajectories
 */
export async function getAvailableNBranches(): Promise<number[]> {
  try {
    const manifest = await loadManifest();
    
    if (manifest.continuous_config?.branching_params?.n_branches) {
      const nBranches = manifest.continuous_config.branching_params.n_branches;
      // Could be a single value or array
      return Array.isArray(nBranches) ? nBranches : [nBranches];
    }
    
    return [2]; // Default
  } catch (error) {
    console.error('Error getting available n_branches:', error);
    return [2];
  }
}

/**
 * Get available n_cycles values for cyclic trajectories
 */
export async function getAvailableNCycles(): Promise<number[]> {
  try {
    const manifest = await loadManifest();
    
    if (manifest.continuous_config?.cyclic_params?.n_cycles) {
      const nCycles = manifest.continuous_config.cyclic_params.n_cycles;
      // Could be a single value or array
      return Array.isArray(nCycles) ? nCycles : [nCycles];
    }
    
    return [1.5]; // Default
  } catch (error) {
    console.error('Error getting available n_cycles:', error);
    return [1.5];
  }
}

/**
 * Get available n_clusters values for discrete trajectories
 */
export async function getAvailableNClusters(): Promise<number[]> {
  try {
    const manifest = await loadManifest();
    
    if (manifest.discrete_config?.n_clusters_list) {
      return manifest.discrete_config.n_clusters_list;
    }
    
    return [5]; // Default
  } catch (error) {
    console.error('Error getting available n_clusters:', error);
    return [5];
  }
}

/**
 * Get available target_trajectory values for discrete trajectories
 */
export async function getAvailableTargetTrajectories(): Promise<TrajectoryType[]> {
  try {
    const manifest = await loadManifest();
    
    if (manifest.discrete_config?.target_trajectories) {
      return manifest.discrete_config.target_trajectories as TrajectoryType[];
    }
    
    return ['linear']; // Default
  } catch (error) {
    console.error('Error getting available target trajectories:', error);
    return ['linear'];
  }
}

/**
 * Get available continuity values for a specific trajectory type
 * Continuous trajectories (linear, branching, cyclic) use continuous_config
 * Discrete trajectories use discrete_config
 */
export async function getAvailableContinuitiesForTrajectory(
  trajectoryType: TrajectoryType
): Promise<number[]> {
  try {
    const manifest = await loadManifest();
    
    if (trajectoryType === 'discrete') {
      // Discrete trajectories use discrete_config continuity levels
      if (manifest.discrete_config?.continuity_levels) {
        return [...manifest.discrete_config.continuity_levels].sort((a, b) => a - b);
      }
    } else {
      // Linear, branching, cyclic use continuous_config
      if (manifest.continuous_config?.continuity_levels) {
        return [...manifest.continuous_config.continuity_levels].sort((a, b) => a - b);
      }
    }
    
    // Fallback
    return trajectoryType === 'discrete' 
      ? [0.0, 0.25, 0.5, 0.75, 1.0]
      : [0.85, 0.90, 0.95, 0.99];
    
  } catch (error) {
    console.error('Error getting continuities for trajectory:', error);
    return trajectoryType === 'discrete' 
      ? [0.0, 0.5, 1.0]
      : [0.85, 0.95, 0.99];
  }
}

/**
 * Find the closest available continuity value for a trajectory type
 */
export async function getClosestContinuity(
  trajectoryType: TrajectoryType,
  targetContinuity: number
): Promise<number> {
  const available = await getAvailableContinuitiesForTrajectory(trajectoryType);
  
  if (available.length === 0) return targetContinuity;
  
  // Find closest match
  let closest = available[0];
  let minDiff = Math.abs(available[0] - targetContinuity);
  
  for (const value of available) {
    const diff = Math.abs(value - targetContinuity);
    if (diff < minDiff) {
      minDiff = diff;
      closest = value;
    }
  }
  
  return closest;
}


// ============================================
// BATCH LOADING
// ============================================

/**
 * Load multiple simulations at once
 */
export async function loadSimulationBatch(
  configurations: Array<{
    trajectoryType: TrajectoryType;
    continuity: number;
    replicate: number;
    nCells?: number;
    n_branches?: number;
    n_cycles?: number;
    n_clusters?: number;
    target_trajectory?: TrajectoryType;
  }>
): Promise<SimulationResult[]> {
  const promises = configurations.map(config =>
    loadSimulation(
      config.trajectoryType,
      config.continuity,
      config.replicate,
      config.nCells,
      config.n_branches,
      config.n_cycles,
      config.n_clusters,
      config.target_trajectory
    )
  );
  
  return Promise.all(promises);
}

// ============================================
// CACHE MANAGEMENT
// ============================================

/**
 * Clear all cached data
 */
export function clearCache(): void {
  manifestCache = null;
  parameterLookupCache = null;
  continuityIndexCache = null;
  metricsSummaryCache = null;
  chunkCache.clear();
}

/**
 * Preload a chunk into cache
 */
export async function preloadChunk(chunkId: number): Promise<void> {
  await loadChunk(chunkId);
}