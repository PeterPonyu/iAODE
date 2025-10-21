
import { getCategoryColor, getCategoryInfo } from '@/lib/formatters';
import { cn } from '@/lib/utils';
import { Info } from 'lucide-react';

interface CategoryLegendProps {
  compact?: boolean;
  showTitle?: boolean;
}

export default function CategoryLegend({ 
  compact = false, 
  showTitle = true 
}: CategoryLegendProps) {
  const categories = ['tiny', 'small', 'medium', 'large'] as const;

  if (compact) {
    return (
      <div className="flex flex-wrap items-center gap-3">
        {showTitle && (
          <span className="flex items-center gap-1 text-xs font-semibold text-[rgb(var(--text-secondary))] transition-colors">
            <Info className="h-3 w-3" />
            Dataset Sizes:
          </span>
        )}
        {categories.map((cat) => {
          const colors = getCategoryColor(cat);
          const info = getCategoryInfo(cat);
          return (
            <div
              key={cat}
              className="flex items-center gap-1.5"
              title={info.description}
            >
              <span
                className={cn(
                  'inline-block w-3 h-3 rounded-full border',
                  colors.bg,
                  colors.border
                )}
              />
              <span className="text-xs font-medium capitalize text-[rgb(var(--text-secondary))] transition-colors">
                {info.label}
              </span>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="p-4 bg-[rgb(var(--info-bg))] border border-[rgb(var(--info-border))] rounded-xl shadow-sm transition-all">
      <div className="flex items-start gap-2 mb-4">
        <Info className="w-5 h-5 mt-0.5 flex-shrink-0 text-[rgb(var(--primary))] transition-colors" />
        <div>
          <h3 className="text-sm font-semibold text-[rgb(var(--info-text-strong))] mb-1 transition-colors">
            Dataset Size Categories
          </h3>
          <p className="text-xs text-[rgb(var(--info-text))] transition-colors">
            Datasets are categorized by their cell count
          </p>
        </div>
      </div>
      
      {/* Vertical layout to prevent overlapping */}
      <div className="space-y-3">
        {categories.map((cat) => {
          const colors = getCategoryColor(cat);
          const info = getCategoryInfo(cat);
          return (
            <div
              key={cat}
              className="flex items-center gap-3"
            >
              <span
                className={cn(
                  'inline-flex items-center justify-center w-10 h-10 rounded-lg border-2 text-base font-bold flex-shrink-0',
                  colors.bg,
                  colors.text,
                  colors.border
                )}
              >
                {info.icon}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold capitalize text-[rgb(var(--text-primary))] transition-colors">
                  {info.label}
                </p>
                <p className="text-xs text-[rgb(var(--text-secondary))] transition-colors">
                  {info.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
