'use client';

import { DatasetStats } from '@/types/datasets';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface CategoryDistributionProps {
  stats: DatasetStats;
}

const CATEGORY_COLORS = {
  tiny: '#9CA3AF',    // gray
  small: '#3B82F6',   // blue
  medium: '#F59E0B',  // yellow/orange
  large: '#10B981',   // green
};

const CATEGORY_LABELS = {
  tiny: 'Tiny (1-5k cells)',
  small: 'Small (5-20k cells)',
  medium: 'Medium (20-50k cells)',
  large: 'Large (50k+ cells)',
};

export default function CategoryDistribution({ stats }: CategoryDistributionProps) {
  const data = Object.entries(stats.categoryDistribution)
    .filter(([_, count]) => count > 0)
    .map(([category, count]) => ({
      name: CATEGORY_LABELS[category as keyof typeof CATEGORY_LABELS] || category,
      value: count,
      percentage: ((count / stats.totalDatasets) * 100).toFixed(1),
      color: CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS] || '#6B7280',
    }))
    .sort((a, b) => b.value - a.value);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="card p-3 shadow-lg">
          <p className="font-medium text-[rgb(var(--foreground))] mb-1 transition-colors">
            {data.name}
          </p>
          <p className="text-sm text-[rgb(var(--muted-foreground))] transition-colors">
            {data.value} datasets ({data.percentage}%)
          </p>
        </div>
      );
    }
    return null;
  };

  const CustomLegend = ({ payload }: any) => {
    return (
      <ul className="flex flex-wrap justify-center gap-4 mt-4">
        {payload.map((entry: any, index: number) => (
          <li key={`legend-${index}`} className="flex items-center gap-2">
            <span 
              className="w-3 h-3 rounded-sm" 
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-sm text-[rgb(var(--text-secondary))] transition-colors">
              {entry.value} ({entry.payload.value})
            </span>
          </li>
        ))}
      </ul>
    );
  };

  return (
    <div className="card p-6">
      <h2 className="text-xl font-bold text-[rgb(var(--foreground))] mb-4 transition-colors">
        Dataset Size Distribution
      </h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="55%"
              labelLine={false}
              label={({ percentage }) => `${percentage}%`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend content={<CustomLegend />} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}