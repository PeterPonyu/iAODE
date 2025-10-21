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
    : { bg: '', text: '', border: '' };

  const info = variant !== 'default' ? getCategoryInfo(variant) : null;
  
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full border-2 text-xs font-semibold transition-all',
        variant === 'default' 
          ? 'bg-[rgb(var(--border-light))] text-[rgb(var(--card-foreground))] border-[rgb(var(--text-muted))]'
          : [colors.bg, colors.text, colors.border],
        className
      )}
      title={info?.description}
    >
      {showLabel && info && <span className="capitalize">{info.label}:</span>}
      {children}
    </span>
  );
}