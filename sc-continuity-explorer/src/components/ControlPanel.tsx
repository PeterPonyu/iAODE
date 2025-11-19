// ============================================================================
// FILE: components/ControlPanel.tsx (FIXED - Conditional Color Options)
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
  const trajectoryTypes: TrajectoryType[] = ['linear', 'branching', 'cyclic', 'discrete'];
  const embeddingMethods: EmbeddingMethod[] = ['pca', 'umap', 'tsne'];
  
  // ✅ FIX: Dynamic color options based on trajectory type
  const getAvailableColorOptions = (): { value: ColorByOption; label: string }[] => {
    const baseOptions = [
      { value: 'pseudotime' as ColorByOption, label: 'Pseudotime' },
      { value: 'cell_types' as ColorByOption, label: 'Cell Types' },
    ];

    // Add trajectory-specific options
    switch (props.trajectoryType) {
      case 'branching':
        baseOptions.push({ value: 'branch_id' as ColorByOption, label: 'Branch ID' });
        break;
      case 'cyclic':
        baseOptions.push({ value: 'cycle_phase' as ColorByOption, label: 'Cycle Phase' });
        break;
      case 'discrete':
        baseOptions.push({ value: 'cluster_labels' as ColorByOption, label: 'Cluster Labels' });
        break;
    }

    return baseOptions;
  };

  const colorOptions = getAvailableColorOptions();

  // ✅ FIX: Reset colorBy if current option is not available for new trajectory
  const handleTrajectoryTypeChange = (newType: TrajectoryType) => {
    props.onTrajectoryTypeChange(newType);
    
    // Check if current colorBy is valid for new trajectory type
    const availableOptions = getAvailableColorOptions();
    const isCurrentValid = availableOptions.some(opt => opt.value === props.colorBy);
    
    if (!isCurrentValid) {
      props.onColorByChange('pseudotime'); // Reset to default
    }
  };

  return (
    <div className="bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg p-6 space-y-6">
      <h2 className="text-lg font-semibold">Controls</h2>

      {/* Trajectory Type */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Trajectory Type</label>
        <select
          value={props.trajectoryType}
          onChange={(e) => handleTrajectoryTypeChange(e.target.value as TrajectoryType)}
          disabled={props.isLoading}
          className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] disabled:opacity-50"
        >
          {trajectoryTypes.map((type) => (
            <option key={type} value={type}>
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {/* Continuity Slider */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">
          Continuity: <strong>{props.continuity.toFixed(3)}</strong>
        </label>
        <input
          type="range"
          min="0"
          max={props.availableContinuities.length - 1}
          step="1"
          value={props.availableContinuities.indexOf(
            props.availableContinuities.find((c) => Math.abs(c - props.continuity) < 0.001) || props.continuity
          )}
          onChange={(e) => props.onContinuityChange(props.availableContinuities[parseInt(e.target.value)])}
          disabled={props.isLoading || props.availableContinuities.length === 0}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-[var(--color-muted-foreground)]">
          <span>{props.availableContinuities[0]?.toFixed(2) || '0.85'}</span>
          <span>{props.availableContinuities[props.availableContinuities.length - 1]?.toFixed(2) || '0.99'}</span>
        </div>
      </div>

      {/* Replicate */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Replicate</label>
        <select
          value={props.replicate}
          onChange={(e) => props.onReplicateChange(parseInt(e.target.value))}
          disabled={props.isLoading}
          className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] disabled:opacity-50"
        >
          {[0, 1, 2].map((rep) => (
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
          className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] disabled:opacity-50"
        >
          {embeddingMethods.map((method) => (
            <option key={method} value={method}>
              {method.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      {/* Color By - ✅ FIXED: Now shows only available options */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Color By</label>
        <select
          value={props.colorBy}
          onChange={(e) => props.onColorByChange(e.target.value as ColorByOption)}
          disabled={props.isLoading}
          className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] disabled:opacity-50"
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
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={props.showMetrics}
          onChange={(e) => props.onShowMetricsChange(e.target.checked)}
          className="w-4 h-4"
        />
        <span className="text-sm font-medium">Show Metrics</span>
      </label>

      {/* Refresh Button */}
      <button
        onClick={props.onRefresh}
        disabled={props.isLoading}
        className="w-full px-4 py-2 bg-[var(--color-primary)] text-[var(--color-primary-foreground)] rounded-lg font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
      >
        {props.isLoading ? 'Loading...' : 'Refresh Data'}
      </button>
    </div>
  );
}
