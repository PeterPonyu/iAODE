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
    
    // Cell range filter
    if (filters.cellRange) {
      const cells = parseInt(dataset.nCells);
      if (isNaN(cells)) return false;
      if (cells < filters.cellRange[0] || cells > filters.cellRange[1]) {
        return false;
      }
    }
    
    // Peak range filter (if added to FilterState)
    if (filters.peakRange) {
      const peaks = parseInt(dataset.nPeaks);
      if (isNaN(peaks)) return false;
      if (peaks < filters.peakRange[0] || peaks > filters.peakRange[1]) {
        return false;
      }
    }
    
    return true;
  });
}

export function getFilterOptions(datasets: MergedDataset[]) {
  const validDatasets = datasets.filter(d => 
    d.nCells !== 'N/A' && d.nPeaks !== 'N/A'
  );
  
  const cellCounts = validDatasets.map(d => parseInt(d.nCells));
  const peakCounts = validDatasets.map(d => parseInt(d.nPeaks));
  
  return {
    organisms: Array.from(new Set(
      datasets.map(d => d.organism).filter(o => o && o !== 'Unknown')
    )).sort(),
    
    platforms: Array.from(new Set(
      datasets.map(d => d.platform).filter(p => p && p !== 'Unknown Platform')
    )).sort(),
    
    categories: ['tiny', 'small', 'medium', 'large'] as const,
    
    cellRange: {
      min: Math.min(...cellCounts),
      max: Math.max(...cellCounts)
    },
    
    peakRange: {
      min: Math.min(...peakCounts),
      max: Math.max(...peakCounts)
    }
  };
}