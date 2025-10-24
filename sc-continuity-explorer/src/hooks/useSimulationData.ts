/**
 * Hook for fetching and managing simulation data
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import { SimulationData, TrajectoryType } from '@/types/simulation';
import { loadSimulation } from '@/lib/dataLoader';

export interface UseSimulationDataOptions {
  trajectoryType: TrajectoryType;
  continuity: number;
  replicate?: number;
  autoLoad?: boolean;
}

export interface UseSimulationDataReturn {
  simulation: SimulationData | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useSimulationData({
  trajectoryType,
  continuity,
  replicate = 1,
  autoLoad = false,
}: UseSimulationDataOptions): UseSimulationDataReturn {
  const [simulation, setSimulation] = useState<SimulationData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSimulationData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await loadSimulation(trajectoryType, continuity, replicate);
      setSimulation(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load simulation';
      setError(message);
      setSimulation(null);
    } finally {
      setIsLoading(false);
    }
  }, [trajectoryType, continuity, replicate]);

  // Auto-load on mount if enabled
  useEffect(() => {
    if (autoLoad) {
      loadSimulationData();
    }
  }, [autoLoad, loadSimulationData]);

  return {
    simulation,
    isLoading,
    error,
    refetch: loadSimulationData,
  };
}