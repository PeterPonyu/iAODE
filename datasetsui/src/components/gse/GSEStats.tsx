
import { GSEGroup } from '@/types/datasets';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { Database, Microscope, BarChart3, HardDrive } from 'lucide-react';

interface GSEStatsProps {
  gseGroup: GSEGroup;
}

export default function GSEStats({ gseGroup }: GSEStatsProps) {
  const { datasets, totalCells, totalPeaks, totalSize } = gseGroup;

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
      label: 'Total Peaks',
      value: formatNumber(totalPeaks),
      icon: BarChart3,
      colorClass: 'text-[rgb(var(--stat-purple))]',
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
              <Icon className={`h-8 w-8 ${stat.colorClass}`} />
            </div>
            <div className="text-2xl font-bold text-[rgb(var(--stat-value))] mb-1">
              {stat.value}
            </div>
            <div className="text-sm text-[rgb(var(--stat-label))]">
              {stat.label}
            </div>
          </div>
        );
      })}
    </div>
  );
}
