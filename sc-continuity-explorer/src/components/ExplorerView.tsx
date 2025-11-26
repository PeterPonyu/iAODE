// ============================================================================
// FILE: components/ExplorerView.tsx
// Main explorer with trajectory-specific continuity handling
// ============================================================================

'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { SimulationResult, TrajectoryType, EmbeddingMethod, ColorByOption } from '@/types/simulation';
import { 
  loadSimulation, 
  getAvailableContinuitiesForTrajectory,
  getClosestContinuity,
  getAvailableTrajectoryTypes,
  getAvailableReplicates,
  getAvailableNBranches,
  getAvailableNCycles,
  getAvailableNClusters,
  getAvailableTargetTrajectories
} from '@/lib/dataLoader';
import { getAvailableEmbeddings } from '@/lib/dataUtils';
import { EmbeddingPlot } from './EmbeddingPlot';
import { MetricsPanel } from './MetricsPanel';
import { ControlPanel } from './ControlPanel';

export function ExplorerView() {
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Base parameters
  const [trajectoryType, setTrajectoryType] = useState<TrajectoryType>('linear');
  const [continuity, setContinuity] = useState(0.95);
  const [replicate, setReplicate] = useState(0);
  
  // Trajectory-specific parameters
  const [nBranches, setNBranches] = useState<number>(2);
  const [nCycles, setNCycles] = useState<number>(1.5);
  const [nClusters, setNClusters] = useState<number>(8);
  const [targetTrajectory, setTargetTrajectory] = useState<TrajectoryType>('linear');
  
  // Visualization parameters
  const [embeddingMethod, setEmbeddingMethod] = useState<EmbeddingMethod>('pca');
  const [colorBy, setColorBy] = useState<ColorByOption>('pseudotime');
  const [showMetrics, setShowMetrics] = useState(true);
  
  // Available options (trajectory-specific)
  const [availableContinuities, setAvailableContinuities] = useState<number[]>([]);
  const [availableTrajectories, setAvailableTrajectories] = useState<TrajectoryType[]>(['linear']);
  const [availableReplicates, setAvailableReplicates] = useState<number[]>([0]);
  const [availableNBranches, setAvailableNBranches] = useState<number[]>([2]);
  const [availableNCycles, setAvailableNCycles] = useState<number[]>([1.5]);
  const [availableNClusters, setAvailableNClusters] = useState<number[]>([8]);
  const [availableTargetTrajectories, setAvailableTargetTrajectories] = useState<TrajectoryType[]>(['linear']);

  // Debounce timer ref for continuity slider
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Load global available options (trajectory-independent)
  useEffect(() => {
    Promise.all([
      getAvailableTrajectoryTypes(),
      getAvailableReplicates(),
      getAvailableNBranches(),
      getAvailableNCycles(),
      getAvailableNClusters(),
      getAvailableTargetTrajectories(),
    ]).then(([trajectories, replicates, nBranchesList, nCyclesList, nClustersList, targetTrajList]) => {
      setAvailableTrajectories(trajectories);
      setAvailableReplicates(replicates);
      setAvailableNBranches(nBranchesList);
      setAvailableNCycles(nCyclesList);
      setAvailableNClusters(nClustersList);
      setAvailableTargetTrajectories(targetTrajList);
      
      // Set default values to first available option
      if (nBranchesList.length > 0) setNBranches(nBranchesList[0]);
      if (nCyclesList.length > 0) setNCycles(nCyclesList[0]);
      if (nClustersList.length > 0) setNClusters(nClustersList[0]);
      if (targetTrajList.length > 0) setTargetTrajectory(targetTrajList[0]);
    }).catch(err => {
      console.error('Failed to load available options:', err);
    });
  }, []);

  // Load trajectory-specific continuity values when trajectory type changes
  useEffect(() => {
    let isMounted = true;
    
    getAvailableContinuitiesForTrajectory(trajectoryType)
      .then(continuities => {
        if (!isMounted) return;
        
        console.log(`Loaded ${continuities.length} continuity values for ${trajectoryType}:`, continuities);
        setAvailableContinuities(continuities);
        
        // Adjust current continuity to closest available for this trajectory
        if (continuities.length > 0) {
          const currentValid = continuities.find(c => Math.abs(c - continuity) < 0.001);
          if (!currentValid) {
            // Find closest match
            getClosestContinuity(trajectoryType, continuity).then(closest => {
              if (!isMounted) return;
              console.log(`Adjusted continuity from ${continuity.toFixed(3)} to ${closest.toFixed(3)} for ${trajectoryType}`);
              setContinuity(closest);
            });
          }
        }
      })
      .catch(err => {
        console.error('Failed to load continuity values:', err);
      });
    
    return () => {
      isMounted = false;
    };
  }, [trajectoryType]); // Only re-run when trajectory type changes

  // Get available embedding methods from loaded simulation
  const availableEmbeddings = useMemo((): EmbeddingMethod[] => {
    return getAvailableEmbeddings(simulation);
  }, [simulation]);

  // Auto-adjust embedding method if current one is not available
  useEffect(() => {
    if (availableEmbeddings.length > 0 && !availableEmbeddings.includes(embeddingMethod)) {
      console.log(`Switching embedding method from ${embeddingMethod} to ${availableEmbeddings[0]}`);
      setEmbeddingMethod(availableEmbeddings[0]);
    }
  }, [availableEmbeddings, embeddingMethod]);

  // Handle trajectory type change with parameter reset
  const handleTrajectoryTypeChange = useCallback((newType: TrajectoryType) => {
    console.log(`Switching trajectory type from ${trajectoryType} to ${newType}`);
    
    // Update trajectory type first
    setTrajectoryType(newType);
    
    // Reset color-by to pseudotime if current option is invalid for new type
    const validColorOptions: ColorByOption[] = ['pseudotime', 'cell_types'];
    if (newType === 'branching') validColorOptions.push('branch_id');
    if (newType === 'cyclic') validColorOptions.push('cycle_phase');
    if (newType === 'discrete') validColorOptions.push('cluster_labels');
    
    if (!validColorOptions.includes(colorBy)) {
      console.log(`Resetting colorBy from ${colorBy} to pseudotime`);
      setColorBy('pseudotime');
    }
    
    // Note: continuity adjustment happens in the useEffect above
  }, [trajectoryType, colorBy]);

  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const params = {
        trajectoryType,
        continuity: continuity.toFixed(3),
        replicate,
        nBranches: trajectoryType === 'branching' ? nBranches : undefined,
        nCycles: trajectoryType === 'cyclic' ? nCycles : undefined,
        nClusters: trajectoryType === 'discrete' ? nClusters : undefined,
        targetTrajectory: trajectoryType === 'discrete' ? targetTrajectory : undefined,
      };
      
      console.log('Loading simulation with params:', params);
      
      const data = await loadSimulation(
        trajectoryType, 
        continuity, 
        replicate,
        500, // n_cells
        trajectoryType === 'branching' ? nBranches : undefined,
        trajectoryType === 'cyclic' ? nCycles : undefined,
        trajectoryType === 'discrete' ? nClusters : undefined,
        trajectoryType === 'discrete' ? targetTrajectory : undefined
      );
      
      console.log('Successfully loaded simulation:', data.id);
      setSimulation(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load simulation';
      console.error('Load error:', message);
      setError(message);
      setSimulation(null);
    } finally {
      setIsLoading(false);
    }
  }, [trajectoryType, continuity, replicate, nBranches, nCycles, nClusters, targetTrajectory]);

  // Debounced version of loadData for slider changes
  const loadDataDebounced = useCallback(() => {
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    // Set new timer - only load after 300ms of no changes
    debounceTimerRef.current = setTimeout(() => {
      loadData();
    }, 300);
  }, [loadData]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  // Only load data when we have valid continuity values
  // Use debounced version for continuity changes
  useEffect(() => {
    if (availableContinuities.length > 0) {
      loadDataDebounced();
    }
  }, [continuity, availableContinuities.length]);

  // Load immediately for non-continuity parameter changes
  useEffect(() => {
    if (availableContinuities.length > 0) {
      loadData();
    }
  }, [trajectoryType, replicate, nBranches, nCycles, nClusters, targetTrajectory, availableContinuities.length]);

  return (
    <div className="w-full min-h-full bg-[rgb(var(--background))]">
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 py-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar */}
          <aside className="lg:w-80 flex-shrink-0">
            <ControlPanel
              trajectoryType={trajectoryType}
              continuity={continuity}
              replicate={replicate}
              nBranches={nBranches}
              nCycles={nCycles}
              nClusters={nClusters}
              targetTrajectory={targetTrajectory}
              embeddingMethod={embeddingMethod}
              colorBy={colorBy}
              showMetrics={showMetrics}
              availableContinuities={availableContinuities}
              availableTrajectories={availableTrajectories}
              availableReplicates={availableReplicates}
              availableNBranches={availableNBranches}
              availableNCycles={availableNCycles}
              availableNClusters={availableNClusters}
              availableTargetTrajectories={availableTargetTrajectories}
              availableEmbeddings={availableEmbeddings}
              onTrajectoryTypeChange={handleTrajectoryTypeChange}
              onContinuityChange={setContinuity}
              onReplicateChange={setReplicate}
              onNBranchesChange={setNBranches}
              onNCyclesChange={setNCycles}
              onNClustersChange={setNClusters}
              onTargetTrajectoryChange={setTargetTrajectory}
              onEmbeddingMethodChange={setEmbeddingMethod}
              onColorByChange={setColorBy}
              onShowMetricsChange={setShowMetrics}
              onRefresh={loadData}
              isLoading={isLoading}
            />
          </aside>

          {/* Main content */}
          <main className="flex-1 min-w-0">
            {error && (
              <div className="mb-6 p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200">
                <strong className="font-semibold">Error:</strong> {error}
                <p className="text-sm mt-2">
                  Try adjusting parameters or check browser console for details.
                </p>
              </div>
            )}

            {isLoading && !simulation && (
              <div className="flex flex-col items-center justify-center min-h-[500px] text-[rgb(var(--muted-foreground))]">
                <div className="w-16 h-16 border-4 border-[rgb(var(--border))] border-t-[rgb(var(--primary))] rounded-full animate-spin mb-4" />
                <p className="text-sm font-medium">Loading simulation data...</p>
                <p className="text-xs mt-1">This may take a few moments</p>
              </div>
            )}

            {!isLoading && !error && !simulation && (
              <div className="flex flex-col items-center justify-center min-h-[500px] text-[rgb(var(--muted-foreground))]">
                <p className="text-sm">Select parameters to load simulation data</p>
              </div>
            )}

            {!error && simulation && (
              <div className="relative space-y-6">
                {/* Loading overlay - show when loading new data but keep old plot visible */}
                {isLoading && (
                  <div className="absolute inset-0 bg-[rgb(var(--background))]/70 backdrop-blur-sm z-10 flex items-center justify-center rounded-lg">
                    <div className="flex flex-col items-center">
                      <div className="w-12 h-12 border-4 border-[rgb(var(--border))] border-t-[rgb(var(--primary))] rounded-full animate-spin mb-2" />
                      <p className="text-sm font-medium text-[rgb(var(--foreground))]">Updating...</p>
                    </div>
                  </div>
                )}
                
                <EmbeddingPlot
                  simulation={simulation}
                  embeddingMethod={embeddingMethod}
                  colorBy={colorBy}
                />

                {showMetrics && (
                  <MetricsPanel
                    metrics={simulation.metrics}
                    parameters={simulation.parameters}
                  />
                )}
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}