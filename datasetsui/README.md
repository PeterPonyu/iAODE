# iAODE Dataset Browser

A web interface for browsing and accessing single-cell ATAC-seq and RNA-seq datasets curated for the iAODE analysis framework.

## Overview

The iAODE Dataset Browser catalogs single-cell genomics datasets organized by GEO Study Accessions (GSE). It functions as a companion tool to the [iAODE package](https://github.com/PeterPonyu/iAODE), facilitating the discovery of datasets suitable for analyzing chromatin accessibility and transcriptional dynamics.

## Features

### Modality Support
- **scATAC-seq**: Chromatin accessibility datasets including peak-level information.
- **scRNA-seq**: Gene expression datasets including transcript counts.

### Dataset Metadata
- **Study Information**: Sourced from NCBI GEO (Accessions, titles, and descriptions).
- **Dataset Specifications**: Cell counts, feature dimensions, species, and tissue types.
- **Quality Metrics**: Pre-computed statistics for dataset evaluation.
- **File Access**: Links to preprocessed `.h5ad` files.

### Interface Functionality
- Filtering and sorting of dataset tables.
- Search functionality across metadata fields.
- Summary statistics for aggregate counts across GSE studies.
- Visual differentiation between modalities (Blue for ATAC-seq, Green for RNA-seq).

## Technology Stack

- **Framework**: Next.js 15 (App Router)
- **UI Library**: React 19
- **Styling**: Tailwind CSS v4
- **Language**: TypeScript
- **Icons**: Lucide React

## Usage

### Browsing Datasets
1. The home page displays summary statistics.
2. Select "Explore scATAC-seq" or "Explore scRNA-seq" to view specific modalities.
3. The list expands to show datasets within GSE studies.
4. Use the search bar and filters to locate specific data.
5. Download links provide access to the preprocessed files.

### Selection Criteria
The datasets are generally prepared with the following thresholds for iAODE analysis:
- **scATAC-seq**: Typically >2,000 cells and >20,000 peaks.
- **scRNA-seq**: Typically >1,000 cells and >15,000 genes.
*Note: Users should verify tissue type and species compatibility for their specific research needs.*

## About iAODE

**iAODE** (Interpretable Accessibility ODE VAE) is a deep learning framework designed for single-cell chromatin accessibility and gene expression analysis. It integrates variational autoencoders and neural ODEs to infer continuous trajectories and extract biological features.

For more information, visit the main repository: [github.com/PeterPonyu/iAODE](https://github.com/PeterPonyu/iAODE).

## Contact

- **Issues**: [GitHub Issues](https://github.com/PeterPonyu/iAODE/issues)
- **Email**: fuzeyu99@126.com