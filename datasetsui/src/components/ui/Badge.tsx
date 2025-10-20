import { cn } from '@/lib/utils';
import { getCategoryColor, getCategoryInfo } from '@/lib/formatters';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'tiny' | 'small' | 'medium' | 'large' | 'error' | 'default';
  className?: string;
  showLabel?: boolean;
}

export function Badge({ 
  children, 
  variant = 'default', 
  className,
  showLabel = false 
}: BadgeProps) {
  const colors = variant !== 'default' 
    ? getCategoryColor(variant)
    : {
        bg: 'bg-gray-200 dark:bg-gray-700',
        text: 'text-gray-800 dark:text-gray-200',
        border: 'border-gray-400 dark:border-gray-500'
      };

  const info = variant !== 'default' ? getCategoryInfo(variant) : null;

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border-2',
        colors.bg,
        colors.text,
        colors.border,
        className
      )}
      title={info?.description}
    >
      {showLabel && info && <span className="capitalize">{info.label}:</span>}
      {children}
    </span>
  );
}