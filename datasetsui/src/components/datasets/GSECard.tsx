import Link from 'next/link';
import { GSEGroup } from '@/types/datasets';
import { Badge } from '@/components/ui/Badge';
import { formatNumber, formatFileSize, getCategoryInfo } from '@/lib/formatters';
import { Database, Microscope, Dna, FileText, Activity } from 'lucide-react';

interface GSECardProps {
  gseGroup: GSEGroup;
  dataType: 'ATAC' | 'RNA';
}

export default function GSECard({ gseGroup, dataType }: GSECardProps) {
  const {
    gseAccession,
    title,
    authors,
    datasets,
    totalCells,
    totalFeatures, // or totalPeaks/totalGenes
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

  // Data type specific styling and labels
  const featureLabel = dataType === 'ATAC' ? 'Peaks' : 'Genes';
  const FeatureIcon = dataType === 'ATAC' ? Activity : Dna;
  
  const cardClasses = dataType === 'ATAC' 
    ? 'card-atac' 
    : 'card-rna';
  
  const accentColor = dataType === 'ATAC'
    ? 'text-[rgb(var(--atac-primary))] group-hover:text-[rgb(var(--atac-primary-hover))]'
    : 'text-[rgb(var(--rna-primary))] group-hover:text-[rgb(var(--rna-primary-hover))]';

  return (
    <Link
      href={`/datasets/${gseAccession}?type=${dataType}`}
      className={`${cardClasses} p-6 hover:shadow-lg transition-all duration-200 flex flex-col h-full group`}
    >
      {/* Header */}
      <div className="mb-3">
        <div className="flex items-start justify-between gap-2 mb-2">
          <h3 className={`font-semibold text-lg transition-colors ${accentColor}`}>
            {gseAccession}
          </h3>
          <div className="flex flex-wrap gap-1.5 justify-end">
            {sortedCategories.map(([cat, count]) => (
              <Badge key={cat} variant={cat as any} showLabel>
                {count}
              </Badge>
            ))}
          </div>
        </div>
        <p className="text-sm text-[rgb(var(--text-secondary))] transition-colors mb-1">
          {authors}
        </p>
      </div>

      {/* Title */}
      <p className="text-sm text-[rgb(var(--card-foreground))] transition-colors mb-4 line-clamp-2 flex-grow">
        {title}
      </p>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 pt-4 border-t border-[rgb(var(--border-light))] transition-colors">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-[rgb(var(--stat-blue))] transition-colors" />
          <div>
            <p className="text-xs text-[rgb(var(--stat-label))] transition-colors font-medium">Datasets</p>
            <p className="text-sm font-semibold text-[rgb(var(--stat-value))] transition-colors">
              {datasets.length}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Microscope className="h-4 w-4 text-[rgb(var(--stat-green))] transition-colors" />
          <div>
            <p className="text-xs text-[rgb(var(--stat-label))] transition-colors font-medium">Cells</p>
            <p className="text-sm font-semibold text-[rgb(var(--stat-value))] transition-colors">
              {formatNumber(totalCells, true)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <FeatureIcon className={`h-4 w-4 transition-colors ${
            dataType === 'ATAC' 
              ? 'text-[rgb(var(--atac-primary))]' 
              : 'text-[rgb(var(--rna-primary))]'
          }`} />
          <div>
            <p className="text-xs text-[rgb(var(--stat-label))] transition-colors font-medium">
              {featureLabel}
            </p>
            <p className="text-sm font-semibold text-[rgb(var(--stat-value))] transition-colors">
              {formatNumber(totalFeatures, true)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-[rgb(var(--stat-orange))] transition-colors" />
          <div>
            <p className="text-xs text-[rgb(var(--stat-label))] transition-colors font-medium">Size</p>
            <p className="text-sm font-semibold text-[rgb(var(--stat-value))] transition-colors">
              {formatFileSize(totalSize)}
            </p>
          </div>
        </div>
      </div>
    </Link>
  );
}