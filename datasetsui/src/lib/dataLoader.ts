import { Dataset, DatasetBrief, MergedDataset, GSEGroup } from '@/types/datasets';
import { extractGsmId, generateDownloadUrl } from './geoUtils';
import { parseNumeric } from './formatters';

// Import all data files
import ATACdatasetsJson from '../data/ATACdatasets.json';
import RNAdatasetsJson from '../data/RNAdatasets.json';
import ATAC_h5_analysisJson from '../data/ATAC_h5_analysis.json';
import RNA_h5_analysisJson from '../data/RNA_h5_analysis.json';

// Separate caches for each data type
let cachedMergedDatasetsATAC: MergedDataset[] | null = null;
let cachedMergedDatasetsRNA: MergedDataset[] | null = null;
let cachedGSEGroupsATAC: GSEGroup[] | null = null;
let cachedGSEGroupsRNA: GSEGroup[] | null = null;

/**
 * Load datasets from JSON file based on data type
 */
export function loadDatasets(dataType: 'ATAC' | 'RNA'): Dataset[] {
  return dataType === 'ATAC' 
    ? (ATACdatasetsJson as Dataset[])
    : (RNAdatasetsJson as Dataset[]);
}

/**
 * Load h5 analysis data from JSON file based on data type
 */
export function loadH5Analysis(dataType: 'ATAC' | 'RNA'): any[] {
  return dataType === 'ATAC'
    ? (ATAC_h5_analysisJson as any[])
    : (RNA_h5_analysisJson as any[]);
}

/**
 * Merge datasets with h5 analysis data
 * Normalizes nPeaks/nGenes to nFeatures
 */
export function mergeDatasets(
  datasets: Dataset[],
  briefData: any[],
  dataType: 'ATAC' | 'RNA'
): MergedDataset[] {
  const briefMap = new Map(
    briefData.map(b => [
      `${b.gseAccession}:${b.dataFileName}`,
      b
    ])
  );

  return datasets.map(dataset => {
    const key = `${dataset.gseAccession}:${dataset.dataFileName}`;
    const brief = briefMap.get(key);

    const gsmId = extractGsmId(dataset.dataFileName);

    // Normalize field names: nPeaks/nGenes → nFeatures
    const nFeatures = dataType === 'ATAC' 
      ? brief?.nPeaks 
      : brief?.nGenes;

    return {
      ...dataset,
      dataFileSize: brief?.dataFileSize || 0,
      nCells: brief?.nCells || 0,
      nFeatures: nFeatures || 0,
      category: brief?.category || 'error',
      gsmId,
      downloadUrl: generateDownloadUrl(gsmId)
    };
  });
}

/**
 * Group datasets by GSE accession
 */
export function groupByGSE(datasets: MergedDataset[]): GSEGroup[] {
  const grouped = new Map<string, MergedDataset[]>();

  datasets.forEach(dataset => {
    const gse = dataset.gseAccession;
    if (!grouped.has(gse)) {
      grouped.set(gse, []);
    }
    grouped.get(gse)!.push(dataset);
  });

  return Array.from(grouped.entries()).map(([gseAccession, datasets]) => {
    // Use first dataset for shared metadata
    const firstDataset = datasets[0];

    // Calculate totals
    const totalCells = datasets.reduce(
      (sum, d) => sum + parseNumeric(d.nCells),
      0
    );
    const totalFeatures = datasets.reduce(
      (sum, d) => sum + parseNumeric(d.nFeatures),
      0
    );
    const totalSize = datasets.reduce(
      (sum, d) => sum + parseNumeric(d.dataFileSize),
      0
    );

    // Get unique platforms
    const platforms = Array.from(
      new Set(datasets.map(d => d.platform).filter(p => p && p !== 'Unknown Platform'))
    );

    return {
      gseAccession,
      title: firstDataset.title,
      authors: firstDataset.authors,
      datasets,
      totalCells,
      totalFeatures,
      totalSize,
      organism: firstDataset.organism,
      platforms
    };
  });
}

/**
 * Get all merged datasets with caching
 */
export function getAllMergedDatasets(dataType: 'ATAC' | 'RNA'): MergedDataset[] {
  const cache = dataType === 'ATAC' ? cachedMergedDatasetsATAC : cachedMergedDatasetsRNA;
  
  if (cache) return cache;
  
  const datasets = loadDatasets(dataType);
  const h5Analysis = loadH5Analysis(dataType);
  const merged = mergeDatasets(datasets, h5Analysis, dataType);
  
  if (dataType === 'ATAC') {
    cachedMergedDatasetsATAC = merged;
  } else {
    cachedMergedDatasetsRNA = merged;
  }
  
  return merged;
}

/**
 * Get all GSE groups with caching
 */
export function getAllGSEGroups(dataType: 'ATAC' | 'RNA'): GSEGroup[] {
  const cache = dataType === 'ATAC' ? cachedGSEGroupsATAC : cachedGSEGroupsRNA;
  
  if (cache) return cache;
  
  const mergedDatasets = getAllMergedDatasets(dataType);
  const groups = groupByGSE(mergedDatasets);
  
  if (dataType === 'ATAC') {
    cachedGSEGroupsATAC = groups;
  } else {
    cachedGSEGroupsRNA = groups;
  }
  
  return groups;
}

/**
 * Get unique GSE accessions
 */
export function getUniqueGSEAccessions(dataType: 'ATAC' | 'RNA'): string[] {
  const datasets = loadDatasets(dataType);
  return Array.from(new Set(datasets.map(d => d.gseAccession)));
}

/**
 * Get GSE group by accession
 */
export function getGSEGroup(gseAccession: string, dataType: 'ATAC' | 'RNA'): GSEGroup | null {
  const allGroups = getAllGSEGroups(dataType);
  return allGroups.find(g => g.gseAccession === gseAccession) || null;
}

/**
 * Get unique organisms
 */
export function getUniqueOrganisms(dataType: 'ATAC' | 'RNA'): string[] {
  const datasets = loadDatasets(dataType);
  return Array.from(new Set(datasets.map(d => d.organism).filter(o => o && o !== 'Unknown')));
}

/**
 * Get unique platforms
 */
export function getUniquePlatforms(dataType: 'ATAC' | 'RNA'): string[] {
  const datasets = loadDatasets(dataType);
  return Array.from(
    new Set(datasets.map(d => d.platform).filter(p => p && p !== 'Unknown Platform'))
  );
}