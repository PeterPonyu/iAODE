/**
 * Overview of key simulation metrics
 */

'use client';

import { SimulationData } from '@/types/simulation';
import { MetricCard } from './MetricCard';
import { QualityIndicators } from './QualityIndicators';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';

export interface MetricsOverviewProps {
  simulation: SimulationData | null;
  className?: string;
}

export function MetricsOverview({ simulation, className = '' }: MetricsOverviewProps) {
  if (!simulation) {
    return (
      <Card className={className}>
        <CardContent className="p-8 text-center text-muted-foreground">
          <p>No simulation data loaded</p>
          <p className="text-sm mt-2">Select parameters and load a simulation to view metrics</p>
        </CardContent>
      </Card>
    );
  }

  const { metadata, parameters, metrics } = simulation;

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Quality Indicators */}
      <Card>
        <CardHeader>
          <CardTitle>Data Quality</CardTitle>
        </CardHeader>
        <CardContent>
          <QualityIndicators
            continuity={parameters.continuity}
            trajectoryType={parameters.trajectory_type}
            cellCount={metadata.n_cells}
          />
        </CardContent>
      </Card>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Cells"
          value={metadata.n_cells}
          format="text"
          icon="🧬"
        />
        <MetricCard
          label="Dimensions"
          value={metadata.n_dims}
          format="text"
          icon="📊"
        />
        <MetricCard
          label="Timepoints"
          value={metadata.n_timepoints}
          format="text"
          icon="⏱️"
        />
        <MetricCard
          label="Continuity"
          value={parameters.continuity}
          format="percent"
          precision={1}
          icon="📈"
        />
      </div>

      {/* Detailed Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Trajectory Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <MetricCard
              label="Mean Branch Length"
              value={metrics.mean_branch_length}
              precision={2}
            />
            <MetricCard
              label="Branch Points"
              value={metrics.n_branch_points}
              format="text"
            />
            <MetricCard
              label="End Points"
              value={metrics.n_end_points}
              format="text"
            />
            <MetricCard
              label="Global Continuity"
              value={metrics.global_continuity}
              format="percent"
              precision={1}
            />
            <MetricCard
              label="Local Continuity"
              value={metrics.local_continuity}
              format="percent"
              precision={1}
            />
            <MetricCard
              label="Trajectory Coverage"
              value={metrics.trajectory_coverage}
              format="percent"
              precision={1}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}