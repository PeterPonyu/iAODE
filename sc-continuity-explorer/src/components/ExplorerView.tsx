// ============================================================================
// FILE: components/ExplorerView.tsx
// Main explorer with integrated state (Tailwind v4 styling)
// ============================================================================

'use client';

import { useState, useEffect, useCallback } from 'react';
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
      <div className="max-w-[1600px] mx-auto p-4 lg:p-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar */}
          <aside className="lg:w-72 flex-shrink-0">
            <ControlPanel
              trajectoryType={trajectoryType}
              continuity={continuity}
              replicate={replicate}
              embeddingMethod={embeddingMethod}
              colorBy={colorBy}
              showMetrics={showMetrics}
              availableContinuities={availableContinuities}
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
              <div className="mb-6 p-4 rounded-lg bg-[var(--color-error)] text-[var(--color-error-foreground)]">
                <strong className="font-semibold">Error:</strong> {error}
              </div>
            )}

            {isLoading && (
              <div className="flex flex-col items-center justify-center min-h-[400px] text-[var(--color-muted-foreground)]">
                <div className="w-10 h-10 border-4 border-[var(--color-border)] border-t-[var(--color-primary)] rounded-full animate-spin mb-4" />
                <p>Loading simulation data...</p>
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
