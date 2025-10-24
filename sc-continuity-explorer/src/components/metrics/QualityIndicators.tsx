/**
 * Data quality indicator badges
 */

'use client';

import { Badge } from '@/components/ui';

export interface QualityIndicatorsProps {
  continuity: number;
  trajectoryType: string;
  cellCount: number;
  className?: string;
}

export function QualityIndicators({
  continuity,
  trajectoryType,
  cellCount,
  className = '',
}: QualityIndicatorsProps) {
  const getContinuityStatus = (value: number) => {
    if (value >= 0.95) return { label: 'Excellent', variant: 'success' as const };
    if (value >= 0.90) return { label: 'Good', variant: 'primary' as const };
    if (value >= 0.85) return { label: 'Fair', variant: 'warning' as const };
    return { label: 'Poor', variant: 'danger' as const };
  };

  const getCellCountStatus = (count: number) => {
    if (count >= 1000) return { label: 'Large Dataset', variant: 'primary' as const };
    if (count >= 500) return { label: 'Medium Dataset', variant: 'default' as const };
    return { label: 'Small Dataset', variant: 'default' as const };
  };

  const continuityStatus = getContinuityStatus(continuity);
  const cellCountStatus = getCellCountStatus(cellCount);

  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      <Badge variant={continuityStatus.variant}>
        {continuityStatus.label} Continuity
      </Badge>
      <Badge variant="default">
        {trajectoryType}
      </Badge>
      <Badge variant={cellCountStatus.variant}>
        {cellCountStatus.label}
      </Badge>
    </div>
  );
}