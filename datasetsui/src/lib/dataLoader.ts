// lib/dataLoader.ts

import { Dataset, DatasetBrief, MergedDataset, GSEGroup } from '@/types/datasets';
import { extractGsmId, generateDownloadUrl } from './geoUtils';
import { parseNumeric } from './formatters';

import datasetsJson from '../data/datasets.json';
import h5AnalysisJson from '../data/h5_analysis.json';

// Add at top of file:
let cachedMergedDatasets: MergedDataset[] | null = null;
let cachedGSEGroups: GSEGroup[] | null = null;

/**
 * Load datasets from JSON file
 */
export function loadDatasets(): Dataset[] {
  return datasetsJson as Dataset[];
}

/**
 * Load h5 analysis data from JSON file
 */
export function loadH5Analysis(): DatasetBrief[] {
  return h5AnalysisJson as DatasetBrief[];
}

/**
 * Merge datasets with h5 analysis data
 */
export function mergeDatasets(
  datasets: Dataset[],
  briefData: DatasetBrief[]
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

    return {
      ...dataset,
      dataFileSize: brief?.dataFileSize || 'N/A',
      nCells: brief?.nCells || 'N/A',
      nPeaks: brief?.nPeaks || 'N/A',
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
    const totalPeaks = datasets.reduce(
      (sum, d) => sum + parseNumeric(d.nPeaks),
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
      totalPeaks,
      totalSize,
      organism: firstDataset.organism,
      platforms
    };
  });
}

// Update getAllMergedDatasets():
export function getAllMergedDatasets(): MergedDataset[] {
  if (cachedMergedDatasets) return cachedMergedDatasets;
  
  const datasets = loadDatasets();
  const h5Analysis = loadH5Analysis();
  cachedMergedDatasets = mergeDatasets(datasets, h5Analysis);
  return cachedMergedDatasets;
}

// Update getAllGSEGroups():
export function getAllGSEGroups(): GSEGroup[] {
  if (cachedGSEGroups) return cachedGSEGroups;
  
  const mergedDatasets = getAllMergedDatasets();
  cachedGSEGroups = groupByGSE(mergedDatasets);
  return cachedGSEGroups;
}

/**
 * Get unique GSE accessions
 */
export function getUniqueGSEAccessions(): string[] {
  const datasets = loadDatasets();
  return Array.from(new Set(datasets.map(d => d.gseAccession)));
}

/**
 * Get GSE group by accession
 */
export function getGSEGroup(gseAccession: string): GSEGroup | null {
  const allGroups = getAllGSEGroups();
  return allGroups.find(g => g.gseAccession === gseAccession) || null;
}

/**
 * Get unique organisms
 */
export function getUniqueOrganisms(): string[] {
  const datasets = loadDatasets();
  return Array.from(new Set(datasets.map(d => d.organism).filter(o => o && o !== 'Unknown')));
}

/**
 * Get unique platforms
 */
export function getUniquePlatforms(): string[] {
  const datasets = loadDatasets();
  return Array.from(
    new Set(datasets.map(d => d.platform).filter(p => p && p !== 'Unknown Platform'))
  );
}