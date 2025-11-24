// ============================================================================
// lib/api.ts - Complete API Functions
// ============================================================================

import { 
  AgentParams, 
  TrainParams, 
  TrainingState, 
  DataInfo, 
  DataType,
  TFIDFParams, 
  HVPParams, 
  SubsampleParams, 
  PreprocessInfo 
} from "./types";

const API_BASE_URL = 'http://localhost:8000';

// ============================================
// DEFAULT PREPROCESSING PARAMETERS
// ============================================

export const DEFAULT_TFIDF_PARAMS: TFIDFParams = {
  scale_factor: 1e4,
  log_tf: false,
  log_idf: true,
};

export const DEFAULT_HVP_PARAMS: HVPParams = {
  n_top_peaks: 20000,
  min_accessibility: 0.01,
  max_accessibility: 0.95,
  method: 'signac',
  use_raw_counts: true,
};

export const DEFAULT_SUBSAMPLE_PARAMS: SubsampleParams = {
  n_cells: undefined,
  frac_cells: undefined,
  use_hvp: true,
  hvp_column: 'highly_variable',
  seed: 42,
};

// ============================================
// DATA UPLOAD
// ============================================

export async function uploadData(file: File, dataType: DataType = 'scrna'): Promise<DataInfo> {
  const formData = new FormData();
  formData.append('file', file);
  
  const url = new URL(`${API_BASE_URL}/upload`);
  url.searchParams.set('data_type', dataType);
  
  const response = await fetch(url.toString(), {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('Failed to upload data');
  }
  
  return response.json();
}

// ============================================
// PREPROCESSING API FUNCTIONS
// ============================================

export async function applyTFIDF(params: TFIDFParams): Promise<PreprocessInfo> {
  const response = await fetch(`${API_BASE_URL}/preprocess/tfidf`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });
  
  if (!response.ok) {
    throw new Error('Failed to apply TF-IDF normalization');
  }
  
  return response.json();
}

export async function selectHVP(params: HVPParams): Promise<PreprocessInfo> {
  const response = await fetch(`${API_BASE_URL}/preprocess/select-hvp`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });
  
  if (!response.ok) {
    throw new Error('Failed to select highly variable peaks');
  }
  
  return response.json();
}

export async function subsampleData(params: SubsampleParams): Promise<PreprocessInfo> {
  const response = await fetch(`${API_BASE_URL}/preprocess/subsample`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });
  
  if (!response.ok) {
    throw new Error('Failed to subsample data');
  }
  
  return response.json();
}

// ============================================
// TRAINING API FUNCTIONS
// ============================================

export async function startTraining(
  agentParams: AgentParams,
  trainParams: TrainParams
): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      agparams: agentParams,
      trainparams: trainParams
    }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to start training');
  }
  
  return response.json();
}

export async function getTrainingState(): Promise<TrainingState> {
  const response = await fetch(`${API_BASE_URL}/state`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch training state');
  }
  
  return response.json();
}

// ============================================
// EMBEDDING DOWNLOAD
// ============================================

export async function downloadEmbedding(embeddingType: 'latent' | 'interpretable'): Promise<void> {
  const url = new URL(`${API_BASE_URL}/download`);
  url.searchParams.set('embedding_type', embeddingType);
  
  const response = await fetch(url.toString());

  if (!response.ok) {
    throw new Error('Failed to download embedding');
  }

  const blob = await response.blob();
  const tempUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = tempUrl;

  // Extract filename from header if provided
  const cd = response.headers.get('content-disposition');
  let filename = `${embeddingType}_embedding.csv`;
  if (cd) {
    const m = cd.match(/filename="(.+)"/);
    if (m?.[1]) filename = m[1];
  }

  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(tempUrl);
}

// ============================================
// STATE MANAGEMENT
// ============================================

export async function resetState(): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}/reset`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error('Failed to reset state');
  }
  
  return response.json();
}