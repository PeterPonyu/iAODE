// ============================================================================
// FILE: components/ControlPanel.tsx
// Control panel with trajectory-specific parameter options
// ============================================================================

'use client';

import { TrajectoryType, EmbeddingMethod, ColorByOption } from '@/types/simulation';

type ControlPanelProps = {
  trajectoryType: TrajectoryType;
  continuity: number;
  replicate: number;
  nBranches: number;
  nCycles: number;
  nClusters: number;
  targetTrajectory: TrajectoryType;
  embeddingMethod: EmbeddingMethod;
  colorBy: ColorByOption;
  showMetrics: boolean;
  availableContinuities: number[];
  availableTrajectories: TrajectoryType[];
  availableReplicates: number[];
  availableNBranches: number[];
  availableNCycles: number[];
  availableNClusters: number[];
  availableTargetTrajectories: TrajectoryType[];
  availableEmbeddings: EmbeddingMethod[];
  onTrajectoryTypeChange: (value: TrajectoryType) => void;
  onContinuityChange: (value: number) => void;
  onReplicateChange: (value: number) => void;
  onNBranchesChange: (value: number) => void;
  onNCyclesChange: (value: number) => void;
  onNClustersChange: (value: number) => void;
  onTargetTrajectoryChange: (value: TrajectoryType) => void;
  onEmbeddingMethodChange: (value: EmbeddingMethod) => void;
  onColorByChange: (value: ColorByOption) => void;
  onShowMetricsChange: (value: boolean) => void;
  onRefresh: () => void;
  isLoading: boolean;
};

export function ControlPanel(props: ControlPanelProps) {
  // Get color options based on trajectory type
  const getColorOptions = (): { value: ColorByOption; label: string }[] => {
    const options = [
      { value: 'pseudotime' as ColorByOption, label: 'Pseudotime' },
      { value: 'cell_types' as ColorByOption, label: 'Cell Types' },
    ];

    if (props.trajectoryType === 'branching') {
      options.push({ value: 'branch_id' as ColorByOption, label: 'Branch ID' });
    } else if (props.trajectoryType === 'cyclic') {
      options.push({ value: 'cycle_phase' as ColorByOption, label: 'Cycle Phase' });
    } else if (props.trajectoryType === 'discrete') {
      options.push({ value: 'cluster_labels' as ColorByOption, label: 'Cluster Labels' });
    }

    return options;
  };

  const colorOptions = getColorOptions();

  const formatLabel = (str: string) => 
    str.charAt(0).toUpperCase() + str.slice(1);

  // Calculate current continuity slider index
  const getCurrentContinuityIndex = (): number => {
    if (props.availableContinuities.length === 0) return 0;
    
    const index = props.availableContinuities.findIndex(
      c => Math.abs(c - props.continuity) < 0.001
    );
    
    return index >= 0 ? index : 0;
  };

  return (
    <div className="bg-[var(--color-background)] border border-[var(--color-border)] rounded-xl p-6 space-y-6 shadow-sm">
      <h2 className="text-lg font-semibold tracking-tight">Controls</h2>

      {/* Trajectory Type */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Trajectory Type</label>
        <select
          value={props.trajectoryType}
          onChange={(e) => props.onTrajectoryTypeChange(e.target.value as TrajectoryType)}
          disabled={props.isLoading}
          className="w-full px-3 py-2.5 text-base border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {props.availableTrajectories.map((type) => (
            <option key={type} value={type}>
              {formatLabel(type)}
            </option>
          ))}
        </select>
      </div>

      {/* ========== TRAJECTORY-SPECIFIC PARAMETERS ========== */}
      
      {/* Branching: n_branches */}
      {props.trajectoryType === 'branching' && props.availableNBranches.length > 1 && (
        <div className="space-y-2">
          <label className="block text-sm font-medium">Number of Branches</label>
          <select
            value={props.nBranches}
            onChange={(e) => props.onNBranchesChange(Number(e.target.value))}
            disabled={props.isLoading}
            className="w-full px-3 py-2.5 text-base border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {props.availableNBranches.map((n) => (
              <option key={n} value={n}>
                {n} branches
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Cyclic: n_cycles */}
      {props.trajectoryType === 'cyclic' && props.availableNCycles.length > 1 && (
        <div className="space-y-2">
          <label className="block text-sm font-medium">Number of Cycles</label>
          <select
            value={props.nCycles}
            onChange={(e) => props.onNCyclesChange(Number(e.target.value))}
            disabled={props.isLoading}
            className="w-full px-3 py-2.5 text-base border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {props.availableNCycles.map((n) => (
              <option key={n} value={n}>
                {n} cycles
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Discrete: n_clusters and target_trajectory */}
      {props.trajectoryType === 'discrete' && (
        <>
          {props.availableNClusters.length > 1 && (
            <div className="space-y-2">
              <label className="block text-sm font-medium">Number of Clusters</label>
              <select
                value={props.nClusters}
                onChange={(e) => props.onNClustersChange(Number(e.target.value))}
                disabled={props.isLoading}
                className="w-full px-3 py-2.5 text-base border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {props.availableNClusters.map((n) => (
                  <option key={n} value={n}>
                    {n} clusters
                  </option>
                ))}
              </select>
            </div>
          )}

          {props.availableTargetTrajectories.length > 1 && (
            <div className="space-y-2">
              <label className="block text-sm font-medium">Target Trajectory</label>
              <select
                value={props.targetTrajectory}
                onChange={(e) => props.onTargetTrajectoryChange(e.target.value as TrajectoryType)}
                disabled={props.isLoading}
                className="w-full px-3 py-2.5 text-base border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {props.availableTargetTrajectories.map((type) => (
                  <option key={type} value={type}>
                    {formatLabel(type)}
                  </option>
                ))}
              </select>
            </div>
          )}
        </>
      )}

      <hr className="border-[var(--color-border)]" />

      {/* Continuity Slider */}
      <div className="space-y-3">
        <label className="block text-sm font-medium">
          Continuity: <span className="font-semibold text-[var(--color-primary)]">{props.continuity.toFixed(3)}</span>
        </label>
        
        {props.availableContinuities.length > 0 ? (
          <>
            <input
              type="range"
              min="0"
              max={Math.max(0, props.availableContinuities.length - 1)}
              step="1"
              value={getCurrentContinuityIndex()}
              onChange={(e) => {
                const index = parseInt(e.target.value);
                const newContinuity = props.availableContinuities[index];
                if (newContinuity !== undefined) {
                  props.onContinuityChange(newContinuity);
                }
              }}
              disabled={props.isLoading}
              className="w-full h-2 bg-[var(--color-muted)] rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <div className="flex justify-between text-xs text-[var(--color-muted-foreground)]">
              <span>{props.availableContinuities[0]?.toFixed(2)}</span>
              <span className="text-center">{props.availableContinuities.length} values</span>
              <span>{props.availableContinuities[props.availableContinuities.length - 1]?.toFixed(2)}</span>
            </div>
          </>
        ) : (
          <div className="text-sm text-[var(--color-muted-foreground)] py-2 text-center">
            Loading continuity options...
          </div>
        )}
      </div>

      {/* Replicate */}
      {props.availableReplicates.length > 1 && (
        <div className="space-y-2">
          <label className="block text-sm font-medium">Replicate</label>
          <select
            value={props.replicate}
            onChange={(e) => props.onReplicateChange(parseInt(e.target.value))}
            disabled={props.isLoading}
            className="w-full px-3 py-2.5 text-base border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {props.availableReplicates.map((rep) => (
              <option key={rep} value={rep}>
                Replicate {rep + 1}
              </option>
            ))}
          </select>
        </div>
      )}

      <hr className="border-[var(--color-border)]" />

      {/* Embedding Method */}
      {props.availableEmbeddings.length > 0 && (
        <div className="space-y-2">
          <label className="block text-sm font-medium">Embedding Method</label>
          <select
            value={props.embeddingMethod}
            onChange={(e) => props.onEmbeddingMethodChange(e.target.value as EmbeddingMethod)}
            disabled={props.isLoading}
            className="w-full px-3 py-2.5 text-base border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {props.availableEmbeddings.map((method) => (
              <option key={method} value={method}>
                {method.toUpperCase()}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Color By */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Color By</label>
        <select
          value={props.colorBy}
          onChange={(e) => props.onColorByChange(e.target.value as ColorByOption)}
          disabled={props.isLoading}
          className="w-full px-3 py-2.5 text-base border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {colorOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <hr className="border-[var(--color-border)]" />

      {/* Show Metrics */}
      <label className="flex items-center gap-3 cursor-pointer group">
        <input
          type="checkbox"
          checked={props.showMetrics}
          onChange={(e) => props.onShowMetricsChange(e.target.checked)}
          className="w-4 h-4 rounded border-[var(--color-border)] text-[var(--color-primary)] cursor-pointer"
        />
        <span className="text-sm font-medium group-hover:text-[var(--color-primary)] transition-colors">
          Show Metrics
        </span>
      </label>

      {/* Refresh Button */}
      <button
        onClick={props.onRefresh}
        disabled={props.isLoading}
        className="w-full px-4 py-2.5 bg-[var(--color-primary)] text-[var(--color-primary-foreground)] rounded-lg font-medium text-sm hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity shadow-sm"
      >
        {props.isLoading ? 'Loading...' : 'Refresh Data'}
      </button>
    </div>
  );
}