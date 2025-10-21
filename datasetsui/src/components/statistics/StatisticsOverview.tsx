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
      iconColor: 'text-[rgb(var(--stat-blue))]',
      bgColor: 'bg-[rgb(var(--stat-bg-blue))]',
    },
    {
      label: 'Total Datasets',
      value: stats.totalDatasets.toString(),
      icon: Database,
      iconColor: 'text-[rgb(var(--stat-purple))]',
      bgColor: 'bg-[rgb(var(--stat-bg-purple))]',
    },
    {
      label: 'Total Cells',
      value: formatNumber(stats.totalCells, true),
      icon: Microscope,
      iconColor: 'text-[rgb(var(--stat-green))]',
      bgColor: 'bg-[rgb(var(--stat-bg-green))]',
      subtitle: `Avg: ${formatNumber(stats.averageCells, true)}/dataset`,
    },
    {
      label: 'Total Peaks',
      value: formatNumber(stats.totalPeaks, true),
      icon: BarChart3,
      iconColor: 'text-[rgb(var(--stat-orange))]',
      bgColor: 'bg-[rgb(var(--stat-bg-orange))]',
      subtitle: `Avg: ${formatNumber(stats.averagePeaks, true)}/dataset`,
    },
    {
      label: 'Total Data Size',
      value: formatFileSize(stats.totalSize),
      icon: HardDrive,
      iconColor: 'text-[rgb(var(--stat-red))]',
      bgColor: 'bg-[rgb(var(--stat-bg-red))]',
      subtitle: `Avg: ${formatFileSize(stats.averageSize)}/dataset`,
    },
    {
      label: 'Median Cells',
      value: formatNumber(stats.medianCells, true),
      icon: TrendingUp,
      iconColor: 'text-[rgb(var(--stat-indigo))]',
      bgColor: 'bg-[rgb(var(--stat-bg-indigo))]',
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
                <p className="text-sm text-[rgb(var(--muted-foreground))] mb-1 transition-colors">
                  {card.label}
                </p>
                <p className="text-2xl font-bold text-[rgb(var(--foreground))] mb-1 transition-colors">
                  {card.value}
                </p>
                {card.subtitle && (
                  <p className="text-xs text-[rgb(var(--muted-foreground))] transition-colors">
                    {card.subtitle}
                  </p>
                )}
              </div>
              <div className={`p-3 rounded-lg ${card.bgColor} transition-colors`}>
                <Icon className={`h-6 w-6 ${card.iconColor} transition-colors`} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
