
// ============================================================================
// FILE: components/ExplorerView.tsx
// Main explorer with fully dynamic options
// ============================================================================

'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { SimulationResult, TrajectoryType, EmbeddingMethod, ColorByOption } from '@/types/simulation';
import { loadSimulation, getAvailableContinuityValues } from '@/lib/dataLoader';
import { EmbeddingPlot } from './EmbeddingPlot';
import { MetricsPanel } from './MetricsPanel';
import { ControlPanel } from './ControlPanel';

export function ExplorerView() {
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [trajectoryType, setTrajectoryType] = useState<TrajectoryType>('linear');
  const [continuity, setContinuity] = useState(0.95);
  const [replicate, setReplicate] = useState(0);
  const [embeddingMethod, setEmbeddingMethod] = useState<EmbeddingMethod>('pca');
  const [colorBy, setColorBy] = useState<ColorByOption>('pseudotime');
  const [showMetrics, setShowMetrics] = useState(true);
  
  const [availableContinuities, setAvailableContinuities] = useState<number[]>([]);

  // Available trajectory types (modify based on your actual data)
  const availableTrajectories: TrajectoryType[] = useMemo(() => {
    // You can make this dynamic by checking manifest or metadata
    return ['linear', 'branching', 'cyclic'];
  }, []);

  // Get available embedding methods from loaded simulation
  const availableEmbeddings = useMemo((): EmbeddingMethod[] => {
    if (!simulation?.embeddings) return ['pca'];
    
    const methods: EmbeddingMethod[] = [];
    if (simulation.embeddings.pca) methods.push('pca');
    if (simulation.embeddings.umap) methods.push('umap');
    if (simulation.embeddings.tsne) methods.push('tsne');
    
    return methods.length > 0 ? methods : ['pca'];
  }, [simulation]);

  // Auto-adjust embedding method if current one is not available
  useEffect(() => {
    if (!availableEmbeddings.includes(embeddingMethod)) {
      setEmbeddingMethod(availableEmbeddings[0]);
    }
  }, [availableEmbeddings, embeddingMethod]);

  useEffect(() => {
    getAvailableContinuityValues().then(setAvailableContinuities).catch(console.error);
  }, []);

  const loadData = useCallback(async () => {
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

  useEffect(() => {
    loadData();
  }, [loadData]);

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 py-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar */}
          <aside className="lg:w-80 flex-shrink-0">
            <ControlPanel
              trajectoryType={trajectoryType}
              continuity={continuity}
              replicate={replicate}
              embeddingMethod={embeddingMethod}
              colorBy={colorBy}
              showMetrics={showMetrics}
              availableContinuities={availableContinuities}
              availableTrajectories={availableTrajectories}
              availableEmbeddings={availableEmbeddings}
              onTrajectoryTypeChange={setTrajectoryType}
              onContinuityChange={setContinuity}
              onReplicateChange={setReplicate}
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
              <div className="mb-6 p-4 rounded-xl bg-[var(--color-error)] text-[var(--color-error-foreground)]">
                <strong className="font-semibold">Error:</strong> {error}
              </div>
            )}

            {isLoading && (
              <div className="flex flex-col items-center justify-center min-h-[500px] text-[var(--color-muted-foreground)]">
                <div className="w-12 h-12 border-4 border-[var(--color-border)] border-t-[var(--color-primary)] rounded-full animate-spin mb-4" />
                <p className="text-sm">Loading simulation data...</p>
              </div>
            )}

            {!isLoading && !error && simulation && (
              <div className="space-y-6">
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
