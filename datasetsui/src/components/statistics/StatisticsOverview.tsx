import { DatasetStats } from '@/types/datasets';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { Database, Beaker, Microscope, BarChart3, HardDrive, TrendingUp } from 'lucide-react';

interface StatisticsOverviewProps {
  stats: DatasetStats;
}

export default function StatisticsOverview({ stats }: StatisticsOverviewProps) {
  const cards = [
    {
      label: 'Total Studies',
      value: stats.totalGSE.toString(),
      icon: Beaker,
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    },
    {
      label: 'Total Datasets',
      value: stats.totalDatasets.toString(),
      icon: Database,
      color: 'text-purple-600 dark:text-purple-400',
      bgColor: 'bg-purple-50 dark:bg-purple-900/20',
    },
    {
      label: 'Total Cells',
      value: formatNumber(stats.totalCells, true),
      icon: Microscope,
      color: 'text-green-600 dark:text-green-400',
      bgColor: 'bg-green-50 dark:bg-green-900/20',
      subtitle: `Avg: ${formatNumber(stats.averageCells, true)}/dataset`,
    },
    {
      label: 'Total Peaks',
      value: formatNumber(stats.totalPeaks, true),
      icon: BarChart3,
      color: 'text-orange-600 dark:text-orange-400',
      bgColor: 'bg-orange-50 dark:bg-orange-900/20',
      subtitle: `Avg: ${formatNumber(stats.averagePeaks, true)}/dataset`,
    },
    {
      label: 'Total Data Size',
      value: formatFileSize(stats.totalSize),
      icon: HardDrive,
      color: 'text-red-600 dark:text-red-400',
      bgColor: 'bg-red-50 dark:bg-red-900/20',
      subtitle: `Avg: ${formatFileSize(stats.averageSize)}/dataset`,
    },
    {
      label: 'Median Cells',
      value: formatNumber(stats.medianCells, true),
      icon: TrendingUp,
      color: 'text-indigo-600 dark:text-indigo-400',
      bgColor: 'bg-indigo-50 dark:bg-indigo-900/20',
      subtitle: 'Per dataset',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <div key={card.label} className="card p-6">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                  {card.label}
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-1">
                  {card.value}
                </p>
                {card.subtitle && (
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {card.subtitle}
                  </p>
                )}
              </div>
              <div className={`p-3 rounded-lg ${card.bgColor}`}>
                <Icon className={`h-6 w-6 ${card.color}`} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}