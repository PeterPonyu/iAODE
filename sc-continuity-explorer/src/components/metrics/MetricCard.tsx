/**
 * Metric display card component
 */

'use client';

import { Card, CardContent } from '@/components/ui';
import { formatNumber, formatPercent } from '@/lib/dataUtils';

export interface MetricCardProps {
  label: string;
  value: string | number | undefined;
  format?: 'text' | 'number' | 'percent';
  precision?: number;
  icon?: string;
  className?: string;
}

export function MetricCard({
  label,
  value,
  format = 'number',
  precision = 2,
  icon,
  className = '',
}: MetricCardProps) {
  // Handle undefined/null values
  const displayValue = getDisplayValue(value, format, precision);
  const isUnavailable = value === undefined || value === null;

  return (
    <Card className={className}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-sm text-muted-foreground mb-1">{label}</p>
            <p className={`text-2xl font-bold ${isUnavailable ? 'text-muted-foreground' : ''}`}>
              {displayValue}
            </p>
          </div>
          {icon && (
            <span className="text-2xl ml-2" role="img" aria-label={label}>
              {icon}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Format value based on type
 */
function getDisplayValue(
  value: string | number | undefined,
  format: 'text' | 'number' | 'percent',
  precision: number
): string {
  // Handle missing values
  if (value === undefined || value === null) {
    return 'N/A';
  }

  // Handle string values
  if (typeof value === 'string') {
    return value;
  }

  // Handle numeric values
  switch (format) {
    case 'percent':
      return formatPercent(value, precision);
    case 'number':
      return formatNumber(value, precision);
    case 'text':
      return String(value);
    default:
      return formatNumber(value, precision);
  }
}