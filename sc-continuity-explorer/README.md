# iAODE Continuity Explorer

An interactive web application for analyzing trajectory structures and continuity metrics across dimensionality reduction methods in single-cell analysis.

## Overview

The iAODE Continuity Explorer allows researchers to investigate how embedding methods (PCA, UMAP, t-SNE) impact trajectory continuity. It provides visualization and quantitative metrics to compare structural preservation across various trajectory types and continuity parameters.

This tool supports the [iAODE package](https://github.com/PeterPonyu/iAODE) by illustrating the effects of dimensionality reduction on trajectory inference and cellular dynamics.

## Features

### Trajectory Types
- **Linear**: Developmental progressions.
- **Branching**: Cell fate decisions and differentiation points.
- **Cyclic**: Periodic processes such as cell cycles.
- **Discrete**: Distinct cell type clusters.

### Embedding Methods
- **PCA**: Linear dimensionality reduction.
- **UMAP**: Manifold learning (global structure preservation).
- **t-SNE**: Stochastic neighbor embedding (local structure preservation).

### Analysis Capabilities
The tool visualizes how continuity parameters influence:
- Trajectory smoothness and connectivity.
- Neighborhood preservation (Local vs. Global).
- Clustering quality.

### Interface
- Visualization of embedding results.
- Quantitative continuity scores.
- Side-by-side method comparison.
- Pre-computed embeddings for efficient data loading.

## Technology Stack

- **Framework**: Next.js 15 (App Router)
- **UI Library**: React 19
- **Styling**: Tailwind CSS v4
- **Visualization**: Recharts
- **Language**: TypeScript

## Usage

### Exploration Workflow
1. **Select Trajectory**: Choose a topology (linear, branching, cyclic, or discrete).
2. **Select Method**: Choose an embedding technique (PCA, UMAP, or t-SNE).
3. **Adjust Parameters**: Use the slider to modify continuity parameters.
4. **Analyze**: Observe changes in the visualization and review the calculated metrics.

### Metrics Definitions
- **Local Preservation**: Measures k-NN consistency between the original and embedded space.
- **Global Structure**: assessed via Distance Correlation (Spearman ρ).
- **Trajectory Coherence**: Evaluates pseudotime consistency along paths.
- **Cluster Separation**: Measured using Silhouette scores and the Davies-Bouldin index.

## Contact

- **Issues**: [GitHub Issues](https://github.com/PeterPonyu/iAODE/issues)
- **Email**: fuzeyu99@126.com