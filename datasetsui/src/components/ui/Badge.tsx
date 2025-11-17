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
  const colors = variant !== 'default' && variant !== 'error'
    ? getCategoryColor(variant)
    : { bg: '', text: '', border: '' };

  const info = variant !== 'default' && variant !== 'error' 
    ? getCategoryInfo(variant) 
    : null;

  // Error variant styling
  if (variant === 'error') {
    return (
      <span
        className={cn(
          'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full border-2 text-xs font-semibold transition-all',
          'bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-400 border-red-300 dark:border-red-800',
          className
        )}
        title="Data parsing error"
      >
        {showLabel && <span>Error:</span>}
        {children}
      </span>
    );
  }

  // Default variant styling
  if (variant === 'default') {
    return (
      <span
        className={cn(
          'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full border-2 text-xs font-semibold transition-all',
          'bg-[rgb(var(--muted))] text-[rgb(var(--foreground))] border-[rgb(var(--border))]',
          className
        )}
      >
        {children}
      </span>
    );
  }
  
  // Category variant styling
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full border-2 text-xs font-semibold transition-all hover:scale-105',
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