# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2025-11-21

### Added

- **Modality-specific examples**: Split examples into scRNA-seq (`*_rna.py`) and scATAC-seq (`*_atac.py`) variants for targeted demonstrations.
- **New comprehensive examples**:
  - `model_evaluation_atac.py`: scATAC-seq benchmarking with iAODE vs scVI-family (Latent Space Evaluation metrics).
  - `model_evaluation_rna.py`: scRNA-seq benchmarking with trajectory-focused analysis.
  - `trajectory_inference_atac.py`: Full scATAC-seq trajectory with multi-panel publication-quality figures.
  - `trajectory_inference_rna.py`: scRNA-seq trajectory with enhanced velocity field visualizations.
- **Enhanced visualizations**:
  - Publication-quality multi-panel figures with 2×2 and 3×3 layouts.
  - Colorblind-friendly palettes (Wong's colors) for model comparisons.
  - Unified color scales across panels (consistent vmin/vmax for fair comparison).
  - Professional typography with Ubuntu font stack.
- **Training progress improvements**:
  - Dual progress bar system: compact main bar with detailed metrics info bar.
  - Real-time metric monitoring (ARI, NMI, ASW, CAL, DAV, COR).
  - Improved time-per-epoch reporting.

### Changed

- **Example file structure**:
  - `basic_usage.py`: Now scATAC-focused with 10X Mouse Brain 5k data, TF-IDF + HVP + MSE loss.
  - `atacseq_annotation.py`: Enhanced QC visualizations with professional color schemes and statistics.
  - Removed legacy unified examples; replaced with modality-specific variants for clarity.
- **Agent training enhancements**:
  - Improved `fit()` method with nested progress bars for better visibility.
  - More robust quiver_autoscale with zero-division guards and render fallbacks.
- **Output organization**:
  - All examples now save to modality-specific subdirectories under `examples/outputs/`.
  - PDF + PNG dual exports for all figures.

### Fixed

- **agent.py**:
  - Fixed quiver_autoscale division-by-zero edge case with proper fallback.
  - Enhanced canvas rendering guards for stable matplotlib interactions.
- **DRE.py**:
  - Fixed Q_local/Q_global key case sensitivity in comprehensive_evaluation method.
- **Version consistency**:
  - Updated LICENSE copyright to 2025.
  - Aligned all version references to 0.2.2.

### Removed

- Legacy unified `model_evaluation.py`, `trajectory_inference.py`, `scATAC_quickstart.py`, `trajectory_ode.py`, `evaluation_metrics.py`.
  - Replaced by modality-specific and more focused examples.

### Notes

- **Focus**: scATAC-seq remains the primary use case with enhanced preprocessing pipelines.
- **Benchmarking**: LSE (Latent Space Evaluation) metrics now standard for all comparisons.
- **Publication readiness**: All new examples produce publication-quality figures with consistent styling.

[0.2.2]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.2.2

## [0.2.1] - 2025-11-20

### Added

- New scATAC-focused examples: `scATAC_quickstart.py`, `trajectory_ode.py`, `evaluation_metrics.py`.
- Refreshed root `README.md` emphasizing scATAC workflow, Neural ODE concept, interpretable bottleneck, recommended hyperparameters.
- Updated `examples/README.md` with curated index and troubleshooting table.
- Added end-to-end scATAC quickstart pipeline (TF-IDF + HVP + training + embeddings export).

### Changed

- Example scripts now use safe fallbacks for `adata.X.copy()` to support multiple AnnData backing modes and satisfy static analysis.
- Improved distance-to-TSS and peak count computations with robust numpy conversions.
- Cast evaluation outputs to native Python types in `DRE.py` and guarded history accesses in `BEN.py`.
- Added defensive ODE tensor casting and attribute guards in `mixin.py`.

### Fixed

- VSCode/Pylance diagnostics in core and example modules (history access, sparse ops, tensor device moves, AnnData matrix copies).
- Potential type issues for dimensionality reduction evaluation by ensuring use of `np.asarray`.

### Removed

- Redundant multi-file benchmarking framework replaced earlier by single streamlined benchmark (retained outside this release scope).

### Notes

- Focus shift: scATAC-seq is now the primary documented use case; legacy scRNA examples retained but marked as legacy.
- Next planned work: logging cleanup, seed consolidation, lightweight logger introduction.

[0.2.1]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.2.1
