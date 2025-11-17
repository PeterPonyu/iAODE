// types/datasets.ts

// General dataset interface
export interface Dataset {
  id: string;
  authors: string;
  title: string;
  gseAccession: string;
  dataFileName: string;
  organism: string;
  source: string;
  platform: string;
}

export interface DatasetBrief {
  gseAccession: string;
  dataFileName: string;
  dataFileSize: number;
  nCells: number;
  nFeatures: number;
  category: 'tiny' | 'small' | 'medium' | 'large' | 'error';
}

export interface MergedDataset extends Dataset {
  dataFileSize: number;
  nCells: number;
  nFeatures: number;
  category: 'tiny' | 'small' | 'medium' | 'large' | 'error';
  gsmId: string;
  downloadUrl: string;
}

export interface GSEGroup {
  gseAccession: string;
  title: string;
  authors: string;
  datasets: MergedDataset[];
  totalCells: number;
  totalFeatures: number;
  totalSize: number;
  organism: string;
  platforms: string[];
}

export interface FilterState {
  search: string;
  categories: ('tiny' | 'small' | 'medium' | 'large' | 'error')[];
  organisms: string[];
  platforms?: string[];
  cellRange: [number, number] | null;
  featureRange: [number, number] | null;
}

export interface DatasetStats {
  totalDatasets: number;
  totalGSE: number;
  totalCells: number;
  totalFeatures: number;
  totalSize: number;
  categoryDistribution: Record<string, number>;
  organismDistribution: Record<string, number>;
  platformDistribution: Record<string, number>;
  averageCells: number;
  averageFeatures: number;
  averageSize: number;
  medianCells: number;
  medianFeatures: number;
}