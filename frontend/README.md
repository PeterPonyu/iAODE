# iAODE Training UI

Interactive web interface for training single-cell RNA-seq and ATAC-seq models using the iAODE (interpretable Adversarial Ordinary Differential Equations) package.

## Features

- **Data Upload**: Support for both scRNA-seq and scATAC-seq data in h5ad format
- **Preprocessing**: TF-IDF normalization, HVP selection, and cell subsampling for scATAC-seq data
- **Model Configuration**: Adjustable hyperparameters for Neural ODE-based trajectory inference
- **Real-time Monitoring**: Live training progress with loss curves and epoch tracking
- **Model Management**: Automatic checkpointing with early stopping

## Getting Started

First, install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to access the training interface.

The UI connects to the FastAPI backend running on port 8000. Use the root-level `start_training_ui.py` or `start_training_ui.sh` scripts to launch both frontend and backend together.

## Technology Stack

- Next.js 15 with App Router
- React 19
- TypeScript
- Tailwind CSS v4
- Recharts for visualization

## About iAODE

iAODE is a Neural ODE-based framework for learning continuous cellular trajectories from single-cell data. The model extracts both latent and interpretable embeddings suitable for downstream analysis like trajectory inference and cell type annotation.
