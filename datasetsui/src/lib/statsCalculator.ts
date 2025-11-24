import { GSEGroup, DatasetStats } from '@/types/datasets';
import { parseNumeric } from './formatters';

export function calculateStats(gseGroups: GSEGroup[], dataType: 'ATAC' | 'RNA'): DatasetStats {
  const allDatasets = gseGroups.flatMap(g => g.datasets);
  
  // Category distribution
  const categoryDistribution = allDatasets.reduce((acc, d) => {
    acc[d.category] = (acc[d.category] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  
  // Organism distribution
  const organismDistribution = allDatasets.reduce((acc, d) => {
    if (d.organism && d.organism !== 'Unknown') {
      acc[d.organism] = (acc[d.organism] || 0) + 1;
    }
    return acc;
  }, {} as Record<string, number>);
  
  // Platform distribution
  const platformDistribution = allDatasets.reduce((acc, d) => {
    if (d.platform && d.platform !== 'Unknown Platform') {
      acc[d.platform] = (acc[d.platform] || 0) + 1;
    }
    return acc;
  }, {} as Record<string, number>);
  
  const totalCells = gseGroups.reduce((sum, g) => sum + g.totalCells, 0);
  const totalFeatures = gseGroups.reduce((sum, g) => sum + g.totalFeatures, 0);
  const totalSize = gseGroups.reduce((sum, g) => sum + g.totalSize, 0);
  
  // Calculate medians
  const cellCounts = allDatasets
    .map(d => parseNumeric(d.nCells))
    .filter(n => n > 0)
    .sort((a, b) => a - b);
  
  const featureCounts = allDatasets
    .map(d => parseNumeric(d.nFeatures))
    .filter(n => n > 0)
    .sort((a, b) => a - b);
  
  const medianCells = cellCounts[Math.floor(cellCounts.length / 2)] || 0;
  const medianFeatures = featureCounts[Math.floor(featureCounts.length / 2)] || 0;
  
  return {
    totalDatasets: allDatasets.length,
    totalGSE: gseGroups.length,
    totalCells,
    totalFeatures,
    totalSize,
    categoryDistribution,
    organismDistribution,
    platformDistribution,
    averageCells: Math.round(totalCells / allDatasets.length),
    averageFeatures: Math.round(totalFeatures / allDatasets.length),
    averageSize: totalSize / allDatasets.length,
    medianCells,
    medianFeatures,
  };
}

// Helper to create histogram bins
export function createHistogramBins(
  values: number[],
  numBins: number = 20
): { bin: string; count: number; min: number; max: number }[] {
  if (values.length === 0) return [];
  
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binSize = (max - min) / numBins;
  
  const bins: { bin: string; count: number; min: number; max: number }[] = [];
  
  for (let i = 0; i < numBins; i++) {
    const binMin = min + i * binSize;
    const binMax = min + (i + 1) * binSize;
    const count = values.filter(v => v >= binMin && v < binMax).length;
    
    // Format bin label
    let binLabel: string;
    if (max > 1000000) {
      binLabel = `${(binMin / 1000000).toFixed(1)}M`;
    } else if (max > 1000) {
      binLabel = `${(binMin / 1000).toFixed(0)}K`;
    } else {
      binLabel = binMin.toFixed(0);
    }
    
    bins.push({
      bin: binLabel,
      count,
      min: binMin,
      max: binMax,
    });
  }
  
  return bins;
}