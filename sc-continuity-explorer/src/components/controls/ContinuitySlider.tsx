/**
 * Continuity parameter slider
 */

'use client';

import { Slider } from '@/components/ui';

export interface ContinuitySliderProps {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  className?: string;
}

export function ContinuitySlider({
  value,
  onChange,
  disabled = false,
  className = '',
}: ContinuitySliderProps) {
  return (
    <Slider
      label="Continuity"
      min={0.8}
      max={1.0}
      step={0.01}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      formatValue={(v) => `${(v * 100).toFixed(0)}%`}
      disabled={disabled}
      className={className}
    />
  );
}