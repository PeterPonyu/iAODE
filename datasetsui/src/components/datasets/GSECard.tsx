
import Link from 'next/link';
import { GSEGroup } from '@/types/datasets';
import { Badge } from '@/components/ui/Badge';
import { formatNumber, formatFileSize, getCategoryInfo } from '@/lib/formatters';
import { Database, Microscope, Dna, FileText } from 'lucide-react';

interface GSECardProps {
  gseGroup: GSEGroup;
}

export default function GSECard({ gseGroup }: GSECardProps) {
  const {
    gseAccession,
    title,
    authors,
    datasets,
    totalCells,
    totalPeaks,
    totalSize,
    organism,
  } = gseGroup;

  // Get category distribution (sorted by category order)
  const categoryOrder = ['tiny', 'small', 'medium', 'large'];
  const categoryCount = datasets.reduce((acc, d) => {
    acc[d.category] = (acc[d.category] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Sort categories
  const sortedCategories = Object.entries(categoryCount)
    .sort(([a], [b]) => categoryOrder.indexOf(a) - categoryOrder.indexOf(b));

  return (
    <Link
      href={`/datasets/${gseAccession}`}
      className="card p-6 hover:shadow-lg transition-shadow duration-200 flex flex-col h-full group"
    >
      {/* Header */}
      <div className="mb-3">
        <div className="flex items-start justify-between gap-2 mb-2">
          <h3 className="font-semibold text-lg text-[rgb(var(--primary))] transition-colors group-hover:text-[rgb(var(--primary-hover))]">
            {gseAccession}
          </h3>
          <div className="flex flex-wrap gap-1.5 justify-end">
            {sortedCategories.map(([cat, count]) => {
              const info = getCategoryInfo(cat);
              return (
                <Badge key={cat} variant={cat as any} showLabel>
                  {count}
                </Badge>
              );
            })}
          </div>
        </div>
        <p className="text-sm text-[rgb(var(--text-secondary))] mb-1">
          {authors}
        </p>
      </div>

      {/* Title */}
      <p className="text-sm text-[rgb(var(--card-foreground))] mb-4 line-clamp-2 flex-grow">
        {title}
      </p>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 pt-4 border-t border-[rgb(var(--border-light))]">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-[rgb(var(--text-tertiary))]" />
          <div>
            <p className="text-xs text-[rgb(var(--text-tertiary))] font-medium">Datasets</p>
            <p className="text-sm font-semibold text-[rgb(var(--text-primary))]">
              {datasets.length}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Microscope className="h-4 w-4 text-[rgb(var(--text-tertiary))]" />
          <div>
            <p className="text-xs text-[rgb(var(--text-tertiary))] font-medium">Cells</p>
            <p className="text-sm font-semibold text-[rgb(var(--text-primary))]">
              {formatNumber(totalCells, true)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Dna className="h-4 w-4 text-[rgb(var(--text-tertiary))]" />
          <div>
            <p className="text-xs text-[rgb(var(--text-tertiary))] font-medium">Organism</p>
            <p className="text-sm font-semibold text-[rgb(var(--text-primary))] truncate">
              {organism}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-[rgb(var(--text-tertiary))]" />
          <div>
            <p className="text-xs text-[rgb(var(--text-tertiary))] font-medium">Size</p>
            <p className="text-sm font-semibold text-[rgb(var(--text-primary))]">
              {formatFileSize(totalSize)}
            </p>
          </div>
        </div>
      </div>
    </Link>
  );
}
