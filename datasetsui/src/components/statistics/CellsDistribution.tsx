'use client';

import { GSEGroup } from '@/types/datasets';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { createHistogramBins } from '@/lib/statsCalculator';
import { parseNumeric, formatNumber } from '@/lib/formatters';

interface CellsDistributionProps {
  gseGroups: GSEGroup[];
}

export default function CellsDistribution({ gseGroups }: CellsDistributionProps) {
  const allDatasets = gseGroups.flatMap(g => g.datasets);
  const cellCounts = allDatasets
    .map(d => parseNumeric(d.nCells))
    .filter(n => n > 0);

  const bins = createHistogramBins(cellCounts, 15);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="card p-3 shadow-lg">
          <p className="font-medium text-gray-900 dark:text-gray-100 mb-1">
            {formatNumber(data.min)} - {formatNumber(data.max)} cells
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {data.count} datasets
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card p-6">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">
        Cell Count Distribution
      </h2>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Distribution of cell counts across all {allDatasets.length} datasets
      </p>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 10, right: 10, left: 10, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
            <XAxis 
              dataKey="bin" 
              className="text-xs fill-gray-600 dark:fill-gray-400"
              label={{ 
                value: 'Cell Count', 
                position: 'insideBottom', 
                offset: -10,
                className: 'fill-gray-600 dark:fill-gray-400'
              }}
            />
            <YAxis 
              className="text-xs fill-gray-600 dark:fill-gray-400"
              label={{ 
                value: 'Number of Datasets', 
                angle: -90, 
                position: 'insideLeft',
                className: 'fill-gray-600 dark:fill-gray-400'
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="count" fill="#3B82F6" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}