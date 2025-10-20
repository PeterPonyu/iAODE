import { getCategoryColor, getCategoryInfo } from '@/lib/formatters';
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
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400 flex items-center gap-1">
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
                className={`
                  inline-block w-3 h-3 rounded-full border
                  ${colors.bg} ${colors.border}
                `}
              />
              <span className="text-xs text-gray-700 dark:text-gray-300 capitalize">
                {info.label}
              </span>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="card p-4 bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-800">
      <div className="flex items-start gap-2 mb-3">
        <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
        <div>
          <h3 className="font-semibold text-sm text-blue-900 dark:text-blue-200 mb-1">
            Dataset Size Categories
          </h3>
          <p className="text-xs text-blue-700 dark:text-blue-300">
            Datasets are categorized by their cell count
          </p>
        </div>
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {categories.map((cat) => {
          const colors = getCategoryColor(cat);
          const info = getCategoryInfo(cat);
          return (
            <div
              key={cat}
              className="flex items-center gap-2"
            >
              <span
                className={`
                  inline-flex items-center justify-center
                  w-8 h-8 rounded-lg border-2 font-bold
                  ${colors.bg} ${colors.text} ${colors.border}
                `}
              >
                {info.icon}
              </span>
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 capitalize">
                  {info.label}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
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