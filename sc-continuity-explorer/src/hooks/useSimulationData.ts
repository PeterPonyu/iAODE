/**
 * Main hook for loading and managing simulation data
 */

'use client';

import { useEffect, useState, useCallback } from 'react';
import { dataLoader } from '@/lib/dataLoader';
import type {
  SimulationResult,
  TrajectoryType,
  DataManifest,
  MetricsSummary,
} from '@/types/simulation';
import { DEFAULTS } from '@/lib/constants';
import { useDebounce } from './useDebounce';

interface UseSimulationDataOptions {
  trajectoryType?: TrajectoryType;
  continuity?: number;
  replicate?: number;
  debounceDelay?: number;
  autoLoad?: boolean;
}

interface UseSimulationDataReturn {
  // Data
  simulation: SimulationResult | null;
  manifest: DataManifest | null;
  metricsSummary: MetricsSummary | null;
  
  // Loading states
  isLoading: boolean;
  isLoadingManifest: boolean;
  error: string | null;
  
  // Actions
  loadSimulation: (
    trajectoryType: TrajectoryType,
    continuity: number,
    replicate?: number
  ) => Promise<void>;
  reload: () => Promise<void>;
  clearError: () => void;
  
  // Current parameters
  currentParams: {
    trajectoryType: TrajectoryType;
    continuity: number;
    replicate: number;
  };
}

/**
 * Hook for loading simulation data with automatic debouncing
 * 
 * @example
 * function ExplorerPage() {
 *   const {
 *     simulation,
 *     isLoading,
 *     error,
 *     loadSimulation,
 *     currentParams,
 *   } = useSimulationData({
 *     trajectoryType: 'linear',
 *     continuity: 0.95,
 *     autoLoad: true,
 *   });
 *   
 *   if (isLoading) return <LoadingSpinner />;
 *   if (error) return <ErrorMessage message={error} />;
 *   if (!simulation) return <EmptyState />;
 *   
 *   return <ScatterPlot data={simulation.embeddings.umap} />;
 * }
 */
export function useSimulationData(
  options: UseSimulationDataOptions = {}
): UseSimulationDataReturn {
  const {
    trajectoryType = DEFAULTS.TRAJECTORY_TYPE,
    continuity = DEFAULTS.CONTINUITY,
    replicate = DEFAULTS.REPLICATE,
    debounceDelay = 300,
    autoLoad = true,
  } = options;

  // State
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [manifest, setManifest] = useState<DataManifest | null>(null);
  const [metricsSummary, setMetricsSummary] = useState<MetricsSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingManifest, setIsLoadingManifest] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Current parameters
  const [currentParams, setCurrentParams] = useState({
    trajectoryType,
    continuity,
    replicate,
  });

  // Debounce continuity to avoid excessive loading during slider drag
  const debouncedContinuity = useDebounce(currentParams.continuity, debounceDelay);

  // Load manifest and metrics summary on mount
  useEffect(() => {
    loadMetadata();
  }, []);

  // Load simulation when debounced parameters change
  useEffect(() => {
    if (autoLoad && manifest) {
      loadSimulation(
        currentParams.trajectoryType,
        debouncedContinuity,
        currentParams.replicate
      );
    }
  }, [
    currentParams.trajectoryType,
    debouncedContinuity,
    currentParams.replicate,
    manifest,
    autoLoad,
  ]);

  /**
   * Load manifest and metrics summary
   */
  const loadMetadata = async () => {
    setIsLoadingManifest(true);
    try {
      const [manifestData, metricsData] = await Promise.all([
        dataLoader.loadManifest(),
        dataLoader.loadMetricsSummary(),
      ]);
      
      setManifest(manifestData);
      setMetricsSummary(metricsData);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load metadata';
      setError(message);
      console.error('Failed to load metadata:', err);
    } finally {
      setIsLoadingManifest(false);
    }
  };

  /**
   * Load a specific simulation
   */
  const loadSimulation = useCallback(
    async (
      trajectoryType: TrajectoryType,
      continuity: number,
      replicate: number = 0
    ) => {
      setIsLoading(true);
      setError(null);

      try {
        // Update current params
        setCurrentParams({ trajectoryType, continuity, replicate });

        // Load simulation
        const data = await dataLoader.loadSimulation(
          trajectoryType,
          continuity,
          replicate
        );

        if (!data) {
          throw new Error('Simulation not found');
        }

        setSimulation(data);

        // Preload adjacent continuity levels for smooth navigation
        dataLoader.preloadAdjacent(trajectoryType, continuity, replicate);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load simulation';
        setError(message);
        console.error('Failed to load simulation:', err);
        setSimulation(null);
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  /**
   * Reload current simulation
   */
  const reload = useCallback(async () => {
    await loadSimulation(
      currentParams.trajectoryType,
      currentParams.continuity,
      currentParams.replicate
    );
  }, [currentParams, loadSimulation]);

  /**
   * Clear error message
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    simulation,
    manifest,
    metricsSummary,
    isLoading,
    isLoadingManifest,
    error,
    loadSimulation,
    reload,
    clearError,
    currentParams,
  };
}

/**
 * Hook for loading multiple simulations (for comparison view)
 */
export function useComparisonData(
  trajectoryType: TrajectoryType,
  continuities: number[],
  replicate: number = 0
) {
  const [simulations, setSimulations] = useState<(SimulationResult | null)[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSimulations();
  }, [trajectoryType, continuities.join(','), replicate]);

  const loadSimulations = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await dataLoader.loadSimulations(
        trajectoryType,
        continuities,
        replicate
      );

      setSimulations(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load simulations';
      setError(message);
      console.error('Failed to load simulations:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    simulations,
    isLoading,
    error,
    reload: loadSimulations,
  };
}