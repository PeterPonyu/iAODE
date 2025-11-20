# Changelog

All notable changes to this project will be documented in this file.

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