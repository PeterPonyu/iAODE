
// ============================================================================
// FILE: components/MetricsPanel.tsx
// Enhanced metrics display
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
    <div className="card space-y-8">
      <h2 className="text-xl font-semibold tracking-tight text-[rgb(var(--text-primary))]">Metrics</h2>
      
      {/* Parameters Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-5 bg-[rgb(var(--muted))] rounded-xl">
        <div>
          <div className="text-xs text-[rgb(var(--muted-foreground))] mb-1 font-medium uppercase tracking-wide">
            Cells
          </div>
          <div className="text-lg font-semibold text-[rgb(var(--text-primary))]">{parameters.n_cells.toLocaleString()}</div>
        </div>
        <div>
          <div className="text-xs text-[rgb(var(--muted-foreground))] mb-1 font-medium uppercase tracking-wide">
            Dimensions
          </div>
          <div className="text-lg font-semibold text-[rgb(var(--text-primary))]">{parameters.n_dims}</div>
        </div>
        <div>
          <div className="text-xs text-[rgb(var(--muted-foreground))] mb-1 font-medium uppercase tracking-wide">
            Continuity
          </div>
          <div className="text-lg font-semibold text-[rgb(var(--text-primary))]">{formatNumber(parameters.continuity, 3)}</div>
        </div>
        <div>
          <div className="text-xs text-[rgb(var(--muted-foreground))] mb-1 font-medium uppercase tracking-wide">
            Replicate
          </div>
          <div className="text-lg font-semibold text-[rgb(var(--text-primary))]">{parameters.replicate + 1}</div>
        </div>
      </div>

      {/* Metrics Groups */}
      {Object.entries(metricGroups).map(([groupName, groupMetrics]) => {
        const availableMetrics = groupMetrics.filter(
          ({ key }) => metrics[key as keyof MetricsData] !== undefined
        );

        if (availableMetrics.length === 0) return null;

        return (
          <div key={groupName}>
            <h3 className="text-xs font-semibold text-[rgb(var(--muted-foreground))] uppercase tracking-wider mb-4">
              {groupName}
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {availableMetrics.map(({ key, label }) => {
                const value = metrics[key as keyof MetricsData];
                return (
                  <div 
                    key={key} 
                    className="flex justify-between items-center px-4 py-3 bg-[rgb(var(--muted))] rounded-lg hover:bg-[rgb(var(--border))] transition-colors"
                  >
                    <span className="text-sm text-[rgb(var(--foreground))] font-medium">
                      {label}
                    </span>
                    <span className="text-sm font-semibold font-mono text-[rgb(var(--primary))]">
                      {formatNumber(value as number, 4)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
