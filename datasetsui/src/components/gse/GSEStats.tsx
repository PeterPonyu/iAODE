import { GSEGroup } from '@/types/datasets';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { Database, Microscope, Activity, Dna, HardDrive } from 'lucide-react';

interface GSEStatsProps {
  gseGroup: GSEGroup;
  dataType: 'ATAC' | 'RNA';
}

export default function GSEStats({ gseGroup, dataType }: GSEStatsProps) {
  const { datasets, totalCells, totalFeatures, totalSize } = gseGroup;

  const featureLabel = dataType === 'ATAC' ? 'Total Peaks' : 'Total Genes';
  const FeatureIcon = dataType === 'ATAC' ? Activity : Dna;
  const featureColorClass = dataType === 'ATAC' 
    ? 'text-[rgb(var(--atac-primary))]' 
    : 'text-[rgb(var(--rna-primary))]';

  const stats = [
    {
      label: 'Total Datasets',
      value: datasets.length.toString(),
      icon: Database,
      colorClass: 'text-[rgb(var(--stat-blue))]',
    },
    {
      label: 'Total Cells',
      value: formatNumber(totalCells),
      icon: Microscope,
      colorClass: 'text-[rgb(var(--stat-green))]',
    },
    {
      label: featureLabel,
      value: formatNumber(totalFeatures),
      icon: FeatureIcon,
      colorClass: featureColorClass,
    },
    {
      label: 'Total Size',
      value: formatFileSize(totalSize),
      icon: HardDrive,
      colorClass: 'text-[rgb(var(--stat-orange))]',
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat) => {
        const Icon = stat.icon;
        return (
          <div
            key={stat.label}
            className="card p-6 text-center"
          >
            <div className="flex justify-center mb-3">
              <Icon className={`h-8 w-8 ${stat.colorClass} transition-colors`} />
            </div>
            <div className="text-2xl font-bold text-[rgb(var(--stat-value))] transition-colors mb-1">
              {stat.value}
            </div>
            <div className="text-sm text-[rgb(var(--stat-label))] transition-colors">
              {stat.label}
            </div>
          </div>
        );
      })}
    </div>
  );
}