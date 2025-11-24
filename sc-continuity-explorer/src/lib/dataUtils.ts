// ============================================================================
// lib/dataUtils.ts - Data Utility Functions
// ============================================================================

import { 
  SimulationResult, 
  EmbeddingMethod, 
  EmbeddingArray,
  Point2D,
  ColorByOption,
  MetricsData,
  Bounds
} from '@/types/simulation';

// ============================================
// NUMBER FORMATTING
// ============================================

/**
 * Format a number with specified decimal places
 * @param value - The number to format
 * @param decimals - Number of decimal places (default: 3)
 * @returns Formatted string or 'N/A' for invalid values
 */
export function formatNumber(value: number | undefined, decimals: number = 3): string {
  if (value === undefined || value === null || isNaN(value) || !isFinite(value)) {
    return 'N/A';
  }
  return value.toFixed(decimals);
}

/**
 * Format metric value for display (alias for backwards compatibility)
 */
export function formatMetricValue(value: number | undefined): string {
  return formatNumber(value, 3);
}

// ============================================
// EMBEDDING UTILITIES
// ============================================

/**
 * Get available embeddings from a simulation result
 */
export function getAvailableEmbeddings(result: SimulationResult | null): EmbeddingMethod[] {
  if (!result) return [];
  
  const embeddings: EmbeddingMethod[] = [];
  
  if (result.embeddings.pca) embeddings.push('pca');
  if (result.embeddings.tsne) embeddings.push('tsne');
  if (result.embeddings.umap) embeddings.push('umap');
  
  return embeddings;
}

/**
 * Get embedding coordinates for a specific method
 */
export function getEmbeddingCoordinates(
  result: SimulationResult,
  method: EmbeddingMethod
): EmbeddingArray | null {
  return result.embeddings[method] || null;
}

/**
 * Get embedding coordinates (alias for backwards compatibility)
 */
export function getEmbedding(
  result: SimulationResult,
  method: EmbeddingMethod
): EmbeddingArray | null {
  return getEmbeddingCoordinates(result, method);
}

/**
 * Get metrics for a specific embedding method
 */
export function getEmbeddingMetrics(
  result: SimulationResult,
  method: EmbeddingMethod
): { variance?: number } {
  const varianceKey = `variance_${method}` as keyof MetricsData;
  const variance = result.metrics[varianceKey];
  
  return {
    variance: typeof variance === 'number' ? variance : undefined
  };
}

// ============================================
// COLOR MAPPING
// ============================================

/**
 * Get color values for points based on the selected color option
 */
export function getColorValues(
  result: SimulationResult,
  colorBy: ColorByOption
): number[] {
  const metadata = result.metadata;
  
  switch (colorBy) {
    case 'pseudotime':
      return metadata.pseudotime;
    
    case 'cell_types':
      // Convert cell types to numeric values
      const uniqueTypes = Array.from(new Set(metadata.cell_types));
      return metadata.cell_types.map(type => uniqueTypes.indexOf(type));
    
    case 'branch_id':
      return metadata.branch_id || [];
    
    case 'cycle_phase':
      return metadata.cycle_phase || [];
    
    case 'cluster_labels':
      return metadata.cluster_labels || [];
    
    default:
      return metadata.pseudotime;
  }
}

/**
 * Get available color-by options for a simulation
 */
export function getAvailableColorOptions(result: SimulationResult): ColorByOption[] {
  const options: ColorByOption[] = ['pseudotime', 'cell_types'];
  
  const metadata = result.metadata;
  
  if (metadata.branch_id && metadata.branch_id.length > 0) {
    options.push('branch_id');
  }
  
  if (metadata.cycle_phase && metadata.cycle_phase.length > 0) {
    options.push('cycle_phase');
  }
  
  if (metadata.cluster_labels && metadata.cluster_labels.length > 0) {
    options.push('cluster_labels');
  }
  
  return options;
}

// ============================================
// COORDINATE TRANSFORMATIONS
// ============================================

/**
 * Calculate bounds of an embedding
 */
export function calculateBounds(coordinates: EmbeddingArray): Bounds {
  if (coordinates.length === 0) {
    return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  }
  
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  
  for (const [x, y] of coordinates) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  
  return { minX, maxX, minY, maxY };
}

/**
 * Normalize coordinates to [0, 1] range
 */
export function normalizeCoordinates(coordinates: EmbeddingArray): EmbeddingArray {
  if (coordinates.length === 0) return [];
  
  const bounds = calculateBounds(coordinates);
  const xRange = bounds.maxX - bounds.minX || 1;
  const yRange = bounds.maxY - bounds.minY || 1;
  
  return coordinates.map(([x, y]): Point2D => [
    (x - bounds.minX) / xRange,
    (y - bounds.minY) / yRange
  ]);
}

/**
 * Scale coordinates to fit within specified dimensions with padding
 */
export function scaleCoordinates(
  coordinates: EmbeddingArray,
  width: number,
  height: number,
  padding: number = 20
): EmbeddingArray {
  if (coordinates.length === 0) return [];
  
  const normalized = normalizeCoordinates(coordinates);
  const availableWidth = width - 2 * padding;
  const availableHeight = height - 2 * padding;
  
  return normalized.map(([x, y]): Point2D => [
    x * availableWidth + padding,
    y * availableHeight + padding
  ]);
}

// ============================================
// METRICS UTILITIES
// ============================================

/**
 * Get core continuity metrics from a simulation
 */
export function getCoreMetrics(result: SimulationResult) {
  const metrics = result.metrics;
  
  return {
    spectral_decay: metrics.spectral_decay,
    anisotropy: metrics.anisotropy,
    participation_ratio: metrics.participation_ratio,
    trajectory_directionality: metrics.trajectory_directionality,
    manifold_dimensionality: metrics.manifold_dimensionality,
    noise_resilience: metrics.noise_resilience
  };
}

// ============================================
// DATA VALIDATION
// ============================================

/**
 * Validate simulation result structure
 */
export function validateSimulationResult(data: any): data is SimulationResult {
  return (
    data &&
    typeof data === 'object' &&
    'id' in data &&
    'parameters' in data &&
    'embeddings' in data &&
    'metrics' in data &&
    'metadata' in data
  );
}

/**
 * Check if a simulation has a specific embedding
 */
export function hasEmbedding(
  result: SimulationResult,
  method: EmbeddingMethod
): boolean {
  return result.embeddings[method] !== undefined;
}

// ============================================
// COMPARISON UTILITIES
// ============================================

/**
 * Compare two simulations and return differences
 */
export function compareSimulations(
  result1: SimulationResult,
  result2: SimulationResult
): {
  metricDifferences: Record<string, number>;
  parameterDifferences: Record<string, [any, any]>;
} {
  const metricDifferences: Record<string, number> = {};
  const parameterDifferences: Record<string, [any, any]> = {};
  
  // Compare metrics
  const allMetricKeys = new Set([
    ...Object.keys(result1.metrics),
    ...Object.keys(result2.metrics)
  ]);
  
  for (const key of allMetricKeys) {
    const val1 = result1.metrics[key as keyof MetricsData] as number | undefined;
    const val2 = result2.metrics[key as keyof MetricsData] as number | undefined;
    
    if (val1 !== undefined && val2 !== undefined) {
      metricDifferences[key] = val2 - val1;
    }
  }
  
  // Compare parameters
  const allParamKeys = new Set([
    ...Object.keys(result1.parameters),
    ...Object.keys(result2.parameters)
  ]);
  
  for (const key of allParamKeys) {
    const val1 = result1.parameters[key as keyof typeof result1.parameters];
    const val2 = result2.parameters[key as keyof typeof result2.parameters];
    
    if (val1 !== val2) {
      parameterDifferences[key] = [val1, val2];
    }
  }
  
  return { metricDifferences, parameterDifferences };
}