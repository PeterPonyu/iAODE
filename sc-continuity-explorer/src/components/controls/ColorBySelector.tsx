'use client';

import { Select, SelectOption } from '@/components/ui';

export interface ColorBySelectorProps {
  value: string;
  onChange: (value: 'pseudotime' | 'cell_types') => void;
  className?: string;
}

export function ColorBySelector({ value, onChange, className = '' }: ColorBySelectorProps) {
  const options: SelectOption[] = [
    { value: 'pseudotime', label: 'Pseudotime' },
    { value: 'cell_types', label: 'Cell Types' },
  ];

  return (
    <Select
      label="Color By"
      value={value}
      onChange={(e) => onChange(e.target.value as 'pseudotime' | 'cell_types')}
      options={options}
      className={className}
    />
  );
}