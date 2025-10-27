
// ============================================================================
// FILE: components/ControlPanel.tsx
// Control panel with all dynamic options
// ============================================================================

'use client';

import { TrajectoryType, EmbeddingMethod, ColorByOption } from '@/types/simulation';

type ControlPanelProps = {
  trajectoryType: TrajectoryType;
  continuity: number;
  replicate: number;
  embeddingMethod: EmbeddingMethod;
  colorBy: ColorByOption;
  showMetrics: boolean;
  availableContinuities: number[];
  availableTrajectories: TrajectoryType[];
  availableReplicates: number[];
  availableEmbeddings: EmbeddingMethod[];
  onTrajectoryTypeChange: (value: TrajectoryType) => void;
  onContinuityChange: (value: number) => void;
  onReplicateChange: (value: number) => void;
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
    }

    return options;
  };

  const colorOptions = getColorOptions();

  // Handle trajectory change with color reset if needed
  const handleTrajectoryChange = (newType: TrajectoryType) => {
    props.onTrajectoryTypeChange(newType);
    
    const isValid = getColorOptions().some(opt => opt.value === props.colorBy);
    if (!isValid) {
      props.onColorByChange('pseudotime');
    }
  };

  const formatLabel = (str: string) => 
    str.charAt(0).toUpperCase() + str.slice(1);

  return (
    <div className="bg-[var(--color-background)] border border-[var(--color-border)] rounded-xl p-6 space-y-6 shadow-sm">
      <h2 className="text-lg font-semibold tracking-tight">Controls</h2>

      {/* Trajectory Type */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Trajectory Type</label>
        <select
          value={props.trajectoryType}
          onChange={(e) => handleTrajectoryChange(e.target.value as TrajectoryType)}
          disabled={props.isLoading}
          className="w-full px-3 py-2 text-sm border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {props.availableTrajectories.map((type) => (
            <option key={type} value={type}>
              {formatLabel(type)}
            </option>
          ))}
        </select>
      </div>

      {/* Continuity Slider */}
      <div className="space-y-3">
        <label className="block text-sm font-medium">
          Continuity: <span className="font-semibold text-[var(--color-primary)]">{props.continuity.toFixed(3)}</span>
        </label>
        <input
          type="range"
          min="0"
          max={Math.max(0, props.availableContinuities.length - 1)}
          step="1"
          value={props.availableContinuities.indexOf(
            props.availableContinuities.find((c) => Math.abs(c - props.continuity) < 0.001) || props.continuity
          )}
          onChange={(e) => props.onContinuityChange(props.availableContinuities[parseInt(e.target.value)])}
          disabled={props.isLoading || props.availableContinuities.length === 0}
          className="w-full h-2 bg-[var(--color-muted)] rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <div className="flex justify-between text-xs text-[var(--color-muted-foreground)]">
          <span>{props.availableContinuities[0]?.toFixed(2) || '0.85'}</span>
          <span>{props.availableContinuities[props.availableContinuities.length - 1]?.toFixed(2) || '0.99'}</span>
        </div>
      </div>

      {/* Replicate - Dynamic */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Replicate</label>
        <select
          value={props.replicate}
          onChange={(e) => props.onReplicateChange(parseInt(e.target.value))}
          disabled={props.isLoading}
          className="w-full px-3 py-2 text-sm border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {props.availableReplicates.map((rep) => (
            <option key={rep} value={rep}>
              Replicate {rep + 1}
            </option>
          ))}
        </select>
      </div>

      <hr className="border-[var(--color-border)]" />

      {/* Embedding Method */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Embedding Method</label>
        <select
          value={props.embeddingMethod}
          onChange={(e) => props.onEmbeddingMethodChange(e.target.value as EmbeddingMethod)}
          disabled={props.isLoading}
          className="w-full px-3 py-2 text-sm border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {props.availableEmbeddings.map((method) => (
            <option key={method} value={method}>
              {method.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      {/* Color By */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Color By</label>
        <select
          value={props.colorBy}
          onChange={(e) => props.onColorByChange(e.target.value as ColorByOption)}
          disabled={props.isLoading}
          className="w-full px-3 py-2 text-sm border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
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
