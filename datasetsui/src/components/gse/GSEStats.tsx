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
      color: 'text-blue-600 dark:text-blue-400',
    },
    {
      label: 'Total Cells',
      value: formatNumber(totalCells),
      icon: Microscope,
      color: 'text-green-600 dark:text-green-400',
    },
    {
      label: 'Total Peaks',
      value: formatNumber(totalPeaks),
      icon: BarChart3,
      color: 'text-purple-600 dark:text-purple-400',
    },
    {
      label: 'Total Size',
      value: formatFileSize(totalSize),
      icon: HardDrive,
      color: 'text-orange-600 dark:text-orange-400',
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
              <Icon className={`h-8 w-8 ${stat.color}`} />
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-1">
              {stat.value}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {stat.label}
            </div>
          </div>
        );
      })}
    </div>
  );
}