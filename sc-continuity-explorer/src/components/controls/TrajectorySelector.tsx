/**
 * Trajectory type selector
 */

'use client';

import { TrajectoryType } from '@/types/simulation';
import { Select, SelectOption } from '@/components/ui';

export interface TrajectorySelectorProps {
  value: TrajectoryType;
  onChange: (value: TrajectoryType) => void;
  disabled?: boolean;
  className?: string;
}

const TRAJECTORY_OPTIONS: SelectOption[] = [
  { value: 'linear', label: '📏 Linear - Simple progression' },
  { value: 'branching', label: '🌳 Branching - Multiple paths' },
  { value: 'cyclic', label: '🔄 Cyclic - Circular pattern' },
  { value: 'discrete', label: '🎯 Discrete - Distinct states' },
];

export function TrajectorySelector({
  value,
  onChange,
  disabled = false,
  className = '',
}: TrajectorySelectorProps) {
  return (
    <Select
      label="Trajectory Type"
      value={value}
      onChange={(e) => onChange(e.target.value as TrajectoryType)}
      options={TRAJECTORY_OPTIONS}
      disabled={disabled}
      helperText="Choose the trajectory pattern for simulation"
      className={className}
    />
  );
}