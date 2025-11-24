// ============================================================================
// lib/types.ts - Complete Type Definitions
// ============================================================================

export type DataType = 'scrna' | 'scatac';

// ============================================
// DATA UPLOAD & INFO
// ============================================

export type DataInfo = {
  n_cells: number;
  n_genes: number;
};

// ============================================
// AGENT & TRAINING PARAMETERS
// ============================================

export type AgentParams = {
  layer: string;
  recon: number;
  irecon: number;
  beta: number;
  dip: number;
  tc: number;
  info: number;
  hidden_dim: number;
  latent_dim: number;
  i_dim: number;
  use_ode: boolean;
  loss_mode: 'mse' | 'nb' | 'zinb';
  lr: number;
  vae_reg: number;
  ode_reg: number;
  train_size: number;
  val_size: number;
  test_size: number;
  batch_size: number;
  random_seed: number;
};

export type TrainParams = {
  epochs: number;
  patience: number;
  val_every: number;
  early_stop: boolean;
};

export type TrainingState = {
  status: string;
  current_epoch: number;
  message: string;
};

// ============================================
// PREPROCESSING PARAMETERS
// ============================================

export type TFIDFParams = {
  scale_factor: number;
  log_tf: boolean;
  log_idf: boolean;
};

export type HVPParams = {
  n_top_peaks: number;
  min_accessibility: number;
  max_accessibility: number;
  method: 'signac' | 'snapatac2' | 'deviance';
  use_raw_counts: boolean;
};

export type SubsampleParams = {
  n_cells?: number;
  frac_cells?: number;
  use_hvp: boolean;
  hvp_column: string;
  seed: number;
};

export type PreprocessInfo = {
  n_cells: number;
  n_peaks: number;
  message: string;
};