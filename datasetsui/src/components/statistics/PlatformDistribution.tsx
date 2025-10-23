'use client';

import { DatasetStats } from '@/types/datasets';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface PlatformDistributionProps {
  stats: DatasetStats;
}

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EF4444', '#6366F1'];

export default function PlatformDistribution({ stats }: PlatformDistributionProps) {
  // Get top 10 platforms
  const data = Object.entries(stats.platformDistribution)
    .map(([platform, count]) => ({
      platform: platform.length > 30 ? platform.substring(0, 30) + '...' : platform,
      fullPlatform: platform,
      count,
      percentage: ((count / stats.totalDatasets) * 100).toFixed(1),
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="card p-3 shadow-lg max-w-xs">
          <p className="font-medium text-gray-900 dark:text-gray-100 mb-1 text-sm">
            {data.fullPlatform}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {data.count} datasets ({data.percentage}%)
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card p-6">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">
        Top Sequencing Platforms
      </h2>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Showing top 10 platforms by dataset count
      </p>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 80 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
            <XAxis 
              dataKey="platform" 
              angle={-45}
              textAnchor="end"
              height={120}
              className="text-xs fill-gray-600 dark:fill-gray-400"
            />
            <YAxis className="text-xs fill-gray-600 dark:fill-gray-400" />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="count" radius={[8, 8, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}