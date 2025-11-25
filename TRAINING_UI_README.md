# iAODE Training UI

Integrated web interface for training iAODE models with real-time monitoring and preprocessing capabilities.

## Features

- 🚀 **Integrated Backend & Frontend**: Single-command startup
- 📊 **Real-time Training Monitor**: Live status updates and metrics
- 🔧 **Preprocessing Pipeline**: TF-IDF, HVP selection, subsampling
- 🎨 **Modern UI**: Dark mode support with polished design
- 📦 **Easy Deployment**: Production-ready static build
- 🔄 **Auto-reload**: Development mode with hot reloading

## Quick Start

### Option 1: Bash Script (Linux/Mac)
```bash
./start_training_ui.sh
```

### Option 2: Python Script (Cross-platform)
```bash
python start_training_ui.py
```

### Option 3: Manual Start
```bash
# 1. Build frontend (first time only)
cd frontend
npm install
npm run build
cd ..

# 2. Start server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Access Points

Once started, access the application at:

- **Training UI**: http://localhost:8000/ui
- **API Documentation**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/

## Architecture

```
iAODE Training UI
├── Backend (FastAPI)
│   ├── REST API endpoints
│   ├── Background training tasks
│   ├── File upload & processing
│   └── Static file serving
│
└── Frontend (Next.js 15)
    ├── Training configuration
    ├── Real-time monitoring
    ├── Preprocessing controls
    └── Embedding download
```

## Configuration

### Backend (API)
- Port: 8000
- CORS: Enabled for localhost:3000 and localhost:8000
- Static files: Served from `frontend/out/`

### Frontend
- Framework: Next.js 15.5.4
- Export: Static HTML/CSS/JS
- Build output: `frontend/out/`
- API endpoint: http://localhost:8000

## API Endpoints

### Training
- `POST /upload` - Upload AnnData file
- `POST /train` - Start training
- `GET /state` - Get training status
- `GET /download` - Download embeddings
- `DELETE /reset` - Reset state

### Preprocessing
- `POST /preprocess/tfidf` - Apply TF-IDF normalization
- `POST /preprocess/select-hvp` - Select highly variable peaks
- `POST /preprocess/subsample` - Subsample cells/peaks

### UI
- `GET /ui` - Training UI homepage
- `GET /ui/train` - Training interface

## Parameters

### Agent Parameters (24 total)
- Architecture: `hidden_dim`, `latent_dim`, `i_dim`, `encoder_type`, `encoder_num_layers`
- Loss weights: `recon`, `irecon`, `beta`, `dip`, `tc`, `info`
- Training: `lr`, `batch_size`, `random_seed`
- Data splits: `train_size`, `val_size`, `test_size`
- ODE: `use_ode`, `vae_reg`, `ode_reg`
- Encoder: `encoder_n_heads`, `encoder_d_model`

### Training Parameters (4 total)
- `epochs` - Maximum training epochs
- `patience` - Early stopping patience
- `val_every` - Validation frequency
- `early_stop` - Enable early stopping

## Development

### Frontend Development
```bash
cd frontend
npm run dev    # Development server on port 3000
npm run build  # Production build
```

### Backend Development
```bash
# With auto-reload
python -m uvicorn api.main:app --reload

# Access interactive docs
# http://localhost:8000/docs
```

### Consistency Check
```bash
python check_consistency.py
```

## Requirements

### Python
- Python 3.8+
- FastAPI
- uvicorn
- iaode package
- scanpy
- anndata
- torch

### Node.js
- Node.js 18+
- npm 9+

## Troubleshooting

### Frontend not showing
```bash
cd frontend && npm run build
```

### CORS errors
Check that CORS middleware in `api/main.py` includes your origin.

### Port already in use
Change port in startup command:
```bash
python -m uvicorn api.main:app --port 8001
```

### Module not found
Install iAODE package:
```bash
pip install -e .
```

## Production Deployment

For production deployment:

1. Build frontend:
```bash
cd frontend && npm run build
```

2. Serve with production ASGI server:
```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

3. Configure reverse proxy (nginx/Apache) for HTTPS.

## License

See main iAODE LICENSE file.
