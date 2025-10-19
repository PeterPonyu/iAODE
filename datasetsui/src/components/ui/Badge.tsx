import { cn } from '@/lib/utils';
import { getCategoryColor } from '@/lib/formatters';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'tiny' | 'small' | 'medium' | 'large' | 'error' | 'default';
  className?: string;
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  const colors = variant !== 'default' 
    ? getCategoryColor(variant)
    : {
        bg: 'bg-gray-100 dark:bg-gray-800',
        text: 'text-gray-700 dark:text-gray-300',
        border: 'border-gray-300 dark:border-gray-600'
      };

  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border',
        colors.bg,
        colors.text,
        colors.border,
        className
      )}
    >
      {children}
    </span>
  );
}