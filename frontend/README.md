# iAODE Training UI

Web-based interface for training iAODE models on single-cell omics data.

## Features

- **Data Upload**: Upload scRNA-seq and scATAC-seq data in H5AD format
- **Preprocessing Pipeline**: TF-IDF normalization, variable peak selection, and subsampling for scATAC-seq data
- **Training Configuration**: Configure model architecture, loss functions, and training parameters
- **Training Monitor**: Real-time monitoring of training progress with early stopping
- **Embedding Export**: Download latent and interpretable embeddings after training

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the training interface.

## Backend

This frontend requires the FastAPI backend to be running. See the main [iAODE](https://github.com/PeterPonyu/iAODE) repository for backend setup instructions.

## Part of iAODE

This project is part of the [iAODE](https://github.com/PeterPonyu/iAODE) framework for interpretable single-cell omics analysis.
