// ============================================================================
// FILE: components/MetricsPanel.tsx
// Metrics display with Tailwind v4
// ============================================================================

'use client';

import { MetricsData, SimulationParameters } from '@/types/simulation';
import { formatNumber } from '@/lib/dataUtils';

type MetricsPanelProps = {
  metrics: MetricsData;
  parameters: SimulationParameters;
};

export function MetricsPanel({ metrics, parameters }: MetricsPanelProps) {
  const metricGroups = {
    'Core Metrics': [
      { key: 'spectral_decay', label: 'Spectral Decay' },
      { key: 'anisotropy', label: 'Anisotropy' },
      { key: 'participation_ratio', label: 'Participation Ratio' },
    ],
    'Trajectory Metrics': [
      { key: 'trajectory_directionality', label: 'Directionality' },
      { key: 'manifold_dimensionality', label: 'Manifold Dimensionality' },
      { key: 'noise_resilience', label: 'Noise Resilience' },
    ],
    'Embedding Variance': [
      { key: 'variance_pca', label: 'PCA' },
      { key: 'variance_umap', label: 'UMAP' },
      { key: 'variance_tsne', label: 't-SNE' },
    ],
  };

  return (
    <div className="bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg p-6 space-y-6">
      <h2 className="text-lg font-semibold">Metrics</h2>
      
      <div className="flex gap-6 p-4 bg-[var(--color-muted)] rounded-lg">
        <div>
          <div className="text-xs text-[var(--color-muted-foreground)]">Cells</div>
          <div className="text-base font-semibold">{parameters.n_cells}</div>
        </div>
        <div>
          <div className="text-xs text-[var(--color-muted-foreground)]">Dimensions</div>
          <div className="text-base font-semibold">{parameters.n_dims}</div>
        </div>
        <div>
          <div className="text-xs text-[var(--color-muted-foreground)]">Continuity</div>
          <div className="text-base font-semibold">{formatNumber(parameters.continuity, 3)}</div>
        </div>
      </div>

      {Object.entries(metricGroups).map(([groupName, groupMetrics]) => (
        <div key={groupName}>
          <h3 className="text-xs font-semibold text-[var(--color-muted-foreground)] uppercase tracking-wider mb-3">
            {groupName}
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {groupMetrics.map(({ key, label }) => {
              const value = metrics[key as keyof MetricsData];
              return value !== undefined ? (
                <div key={key} className="flex justify-between items-center px-3 py-2 bg-[var(--color-muted)] rounded">
                  <span className="text-sm text-[var(--color-muted-foreground)]">{label}</span>
                  <span className="text-sm font-semibold font-mono">{formatNumber(value, 4)}</span>
                </div>
              ) : null;
            })}
          </div>
        </div>
      ))}
    </div>
  );
}