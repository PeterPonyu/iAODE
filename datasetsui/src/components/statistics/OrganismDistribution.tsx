'use client';

import { DatasetStats } from '@/types/datasets';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface OrganismDistributionProps {
  stats: DatasetStats;
}

const ORGANISM_COLORS = [
  '#3B82F6', // blue
  '#10B981', // green
  '#F59E0B', // orange
  '#8B5CF6', // purple
  '#EF4444', // red
  '#6366F1', // indigo
  '#EC4899', // pink
  '#14B8A6', // teal
];

export default function OrganismDistribution({ stats }: OrganismDistributionProps) {
  const data = Object.entries(stats.organismDistribution)
    .map(([organism, count]) => ({
      organism,
      count,
      percentage: ((count / stats.totalDatasets) * 100).toFixed(1),
    }))
    .sort((a, b) => b.count - a.count);

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: { organism: string; count: number; percentage: string } }> }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="card p-3 shadow-lg">
          <p className="font-medium text-[rgb(var(--foreground))] mb-1 transition-colors">
            {data.organism}
          </p>
          <p className="text-sm text-[rgb(var(--muted-foreground))] transition-colors">
            {data.count} datasets ({data.percentage}%)
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card p-6">
      <h2 className="text-xl font-bold text-[rgb(var(--foreground))] mb-4 transition-colors">
        Organism Distribution
      </h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
            <XAxis 
              dataKey="organism" 
              angle={-45}
              textAnchor="end"
              height={100}
              className="text-xs fill-gray-600 dark:fill-gray-400"
            />
            <YAxis className="text-xs fill-gray-600 dark:fill-gray-400" />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="count" radius={[8, 8, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={ORGANISM_COLORS[index % ORGANISM_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}