# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.8] - 2025-11-25

### Changed

- **Training display**: Improved training progress output with cleaner formatting
  - Unified progress bar with tabular metric display
  - Better metric organization and readability
  - Enhanced tqdm integration for notebook compatibility

### Fixed

- **Data organization**: Removed duplicate data directories
  - Cleaned up `data/` and `notebooks/data/` duplicates
  - Kept only `examples/data/` for centralized example datasets
  - Updated `.gitignore` to prevent future data directory duplication

### Documentation

- **Examples README**: Fixed LaTeX notation and added missing example documentation
- **Code documentation**: Improved inline comments and docstrings

[0.2.8]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.2.8

---

## [0.2.7] - 2025-11-23

### Added

- **Jupyter Notebook Examples**: Converted all 6 Python example scripts to interactive Jupyter notebooks (`.ipynb` format) in a new `/notebooks` directory
  - 01_basic_usage.ipynb: scATAC-seq dimensionality reduction
  - 02_atacseq_annotation.ipynb: Peak annotation pipeline
  - 03_trajectory_inference_rna.ipynb: RNA trajectory inference with Neural ODE
  - 04_trajectory_inference_atac.ipynb: ATAC trajectory inference with Neural ODE
  - 05_model_evaluation_rna.ipynb: RNA model benchmarking
  - 06_model_evaluation_atac.ipynb: ATAC model benchmarking
- **Notebooks README**: Comprehensive guide for using the Jupyter notebook examples

### Fixed

- **Data loss handling**: Separate log-transformed and raw count data paths for NB/ZINB loss modes
- **Notebook outputs**: Cleared execution outputs to remove personal file paths for better portability
- **Code completeness**: Verified all Jupyter notebooks contain complete code from original Python files

### Changed

- **Example formats**: Both Python scripts and Jupyter notebooks now available for all examples
- **Notebook outputs**: Clean notebooks ready for execution without personal directory paths

### Notes

- All notebooks verified to contain 100% of code from source Python files
- Notebooks use relative paths for portability across different systems
- Execution outputs cleared for cleaner distribution

[0.2.7]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.2.7

---

## [0.2.4] - 2025-11-22

### Added

- **README badges**: Added PyTorch (>=1.10) and Tests workflow badges for better visibility
- **CI improvements**:
  - Test workflow now properly executes pytest instead of just import checks
  - Added `tomli` dependency to publish workflow for version verification
  - Install dev dependencies in test workflow for proper test execution

### Changed

- **PyPI badge**: Updated to link directly to PyPI project page instead of badge.fury.io
- **Workflow refinements**:
  - Removed verbose package listing from publish workflow (streamlined)
  - Removed Test PyPI publishing step (no token configured)
  - Badge now correctly points to test workflow instead of publish workflow

### Fixed

- **Test execution**: Tests folder now properly runs in CI with pytest
- **Documentation formatting**: Improved examples README with better code block spacing

### Notes

- Tests folder validated as necessary for quality assurance
- All workflows now lean and functional

[0.2.4]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.2.4

---

## [0.2.3] - 2025-11-21

### Changed

- **Example code formatting improvements**:
  - Comprehensive refactoring of all example scripts for consistency and readability
  - Improved docstrings and comments with better descriptions
  - Enhanced code formatting: proper line breaks, consistent indentation, multi-line function calls
  - Unified font stack (Ubuntu font priority) across all visualization examples
  - Streamlined CLI output: simplified print statements, improved table layout
  - Better variable naming for clarity (e.g., `vfres` → `vfres`, `lim_x` → consistent usage)

- **Examples README documentation overhaul**:
  - Complete restructuring with clear sections (Prerequisites, Running, Shared Utilities, Example Overview)
  - Added comprehensive 4.1–4.4 section describing each example's workflow, key steps, figures, and outputs
  - Enhanced pedagogical structure: prerequisites, data download, execution, output organization
  - Improved table of contents with links to specific sections
  - Added LaTeX math formatting for technical descriptions (e.g., pseudotime correlations)
  - Better emphasis on modality-specific workflows (scRNA-seq vs scATAC-seq)

- **Output and visualization consistency**:
  - Panel labels: lowercase titles (e.g., "A. Pseudotime" → "A. pseudotime")
  - Simplified metric legends and colorbar labeling
  - Improved axis label formatting (no unnecessary capitalization)
  - Enhanced visual hierarchy in multi-panel figures

- **Code organization improvements**:
  - Removed redundant comments and developer notes
  - Cleaner function parameter formatting
  - Better separation of logical blocks with section dividers (──────)
  - More consistent error handling and fallback patterns

### Notes

- All example scripts remain functionally identical; changes are cosmetic and organizational.
- Documentation improvements aim to make examples more accessible to new users.
- Ready for publication as stable 0.2.3 release.

[0.2.3]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.2.3

---

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

---

## [0.2.1] - 2025-11-20

### Added (v0.2.1)

- New scATAC-focused examples: `scATAC_quickstart.py`, `trajectory_ode.py`, `evaluation_metrics.py`.
- Refreshed root `README.md` emphasizing scATAC workflow, Neural ODE concept, interpretable bottleneck, recommended hyperparameters.
- Updated `examples/README.md` with curated index and troubleshooting table.
- Added end-to-end scATAC quickstart pipeline (TF-IDF + HVP + training + embeddings export).

### Changed (v0.2.1)

- Example scripts now use safe fallbacks for `adata.X.copy()` to support multiple AnnData backing modes and satisfy static analysis.
- Improved distance-to-TSS and peak count computations with robust numpy conversions.
- Cast evaluation outputs to native Python types in `DRE.py` and guarded history accesses in `BEN.py`.
- Added defensive ODE tensor casting and attribute guards in `mixin.py`.

### Fixed (v0.2.1)

- VSCode/Pylance diagnostics in core and example modules (history access, sparse ops, tensor device moves, AnnData matrix copies).
- Potential type issues for dimensionality reduction evaluation by ensuring use of `np.asarray`.

### Removed (v0.2.1)

- Redundant multi-file benchmarking framework replaced earlier by single streamlined benchmark (retained outside this release scope).

### Notes (v0.2.1)

- Focus shift: scATAC-seq is now the primary documented use case; legacy scRNA examples retained but marked as legacy.
- Next planned work: logging cleanup, seed consolidation, lightweight logger introduction.

[0.2.1]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.2.1

---

## [0.2.0] - 2025-11-20

### Documentation & Usability

- Comprehensive documentation overhaul (README, examples/README, QUICKSTART)
- Separated scRNA-seq and scATAC-seq workflows with clear examples
- Added automatic dataset download and caching system
- Created `examples/data/download_data.sh` helper script
- Verified all API usage examples against actual implementation

### Code Quality

- Cleaned example scripts: removed unused imports, improved error handling
- Fixed all cross-references between examples and core modules
- Enhanced inline documentation and error messages
- Verified method signatures: `get_velocity()`, `get_vfres()`, `get_pseudotime()`, etc.

### Package Improvements

- Complete scATAC-seq preprocessing pipeline with TF-IDF and HVP selection
- Improved peak annotation with distance-to-TSS computation
- Enhanced evaluation metrics (DRE + LSE frameworks)
- Better handling of sparse data and zero inflation

[0.2.0]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.2.0

---

## [0.1.2] - 2025-11-19

### Metadata & Contact

- Updated author name to "Zeyu Fu"
- Set primary contact email to <fuzeyu99@126.com>
- Fixed BibTeX citation formatting

[0.1.2]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.1.2

---

## [0.1.1] - 2025-11-19

### Bug Fixes

- Added missing `requests` dependency for scvi-tools compatibility
- Dropped Python 3.8 support (requires Python ≥3.9)
- Fixed CI test imports and matrix configuration

[0.1.1]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.1.1

---

## [0.1.0] - 2025-11-19

### Initial Release

- VAE with Neural ODE support for trajectory inference
- Complete scATAC-seq peak annotation pipeline
- Comprehensive evaluation metrics
- Benchmark framework vs scVI models
- Multiple encoder types (MLP, Residual, Transformer, Linear)
- Multiple loss modes (MSE, NB, ZINB)

[0.1.0]: https://github.com/PeterPonyu/iAODE/releases/tag/v0.1.0
