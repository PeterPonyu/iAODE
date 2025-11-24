import { MergedDataset, FilterState } from '@/types/datasets';

export function applyFilters(
  datasets: MergedDataset[],
  filters: FilterState
): MergedDataset[] {
  return datasets.filter(dataset => {
    // Category filter
    if (filters.categories.length > 0 && 
        !filters.categories.includes(dataset.category)) {
      return false;
    }
    
    // Organism filter
    if (filters.organisms.length > 0 && 
        !filters.organisms.includes(dataset.organism)) {
      return false;
    }
    
    // Platform filter
    if (filters.platforms && filters.platforms.length > 0 && 
        !filters.platforms.includes(dataset.platform)) {
      return false;
    }
    
    // Cell range filter
    if (filters.cellRange) {
      if (dataset.nCells < filters.cellRange[0] || dataset.nCells > filters.cellRange[1]) {
        return false;
      }
    }
    
    // Feature range filter
    if (filters.featureRange) {
      if (dataset.nFeatures < filters.featureRange[0] || dataset.nFeatures > filters.featureRange[1]) {
        return false;
      }
    }
    
    return true;
  });
}

export function getFilterOptions(datasets: MergedDataset[]) {
  // nCells and nFeatures are always numbers, no need to filter
  const cellCounts = datasets.map(d => d.nCells).filter(n => n > 0);
  const featureCounts = datasets.map(d => d.nFeatures).filter(n => n > 0);
  
  return {
    organisms: Array.from(new Set(
      datasets.map(d => d.organism).filter(o => o && o !== 'Unknown')
    )).sort(),
    
    platforms: Array.from(new Set(
      datasets.map(d => d.platform).filter(p => p && p !== 'Unknown Platform')
    )).sort(),
    
    categories: ['tiny', 'small', 'medium', 'large'] as const,
    
    cellRange: cellCounts.length > 0 ? {
      min: Math.min(...cellCounts),
      max: Math.max(...cellCounts)
    } : null,
    
    featureRange: featureCounts.length > 0 ? {
      min: Math.min(...featureCounts),
      max: Math.max(...featureCounts)
    } : null
  };
}