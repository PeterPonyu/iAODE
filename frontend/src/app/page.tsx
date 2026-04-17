
'use client';

import Link from 'next/link';
import { Header } from '@/components/header';
import { Sparkles, Zap, Eye } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      {/* Hero Section */}
      <main className="flex-1 container mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center space-y-8">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[rgb(var(--training-bg))] text-[rgb(var(--training-text-bright))] text-sm font-medium mb-4">
            <Sparkles className="w-4 h-4" />
            <span>Local-First Training Workspace</span>
          </div>

          <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-[rgb(var(--foreground))]">
            <span className="bg-linear-to-r from-[rgb(var(--training-primary))] to-[rgb(var(--primary))] bg-clip-text text-transparent">
              iAODE Workspace
            </span>
          </h1>
          
          <p className="text-xl max-w-2xl mx-auto text-[rgb(var(--text-secondary))]">
            This interface is designed for local training workflows backed by the iAODE FastAPI service.
            For public datasets, explorers, and project overview pages, use the public iAODE Pages surface or SCPortal.
          </p>

          <div className="flex gap-4 justify-center">
            <Link
              href="/train"
              className="btn-training px-8 py-3 rounded-lg font-medium"
            >
              Run Locally
            </Link>
            <a
              href="https://peterponyu.github.io/"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary px-8 py-3 rounded-lg font-medium"
            >
              Homepage
            </a>
            <a
              href="https://peterponyu.github.io/iAODE/"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary px-8 py-3 rounded-lg font-medium"
            >
              iAODE Pages
            </a>
            <a
              href="https://peterponyu.github.io/scportal/"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary px-8 py-3 rounded-lg font-medium"
            >
              SCPortal
            </a>
          </div>

          <div className="mx-auto max-w-3xl rounded-xl border border-amber-300/40 bg-amber-50/80 px-5 py-4 text-sm text-amber-900 dark:border-amber-700/40 dark:bg-amber-950/30 dark:text-amber-200">
            This site is not part of the public GitHub Pages graph as an interactive training surface.
            It expects a local backend at <code>localhost:8000</code> and is best treated as a local workspace or demo shell.
          </div>
        </div>

        {/* Features */}
        <div className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="card p-6 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 rounded-lg bg-[rgb(var(--info-bg))] text-[rgb(var(--info-text))] flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold mb-2 text-[rgb(var(--foreground))]">Multi-modal Support</h3>
              <p className="text-[rgb(var(--text-secondary))]">
                Work with scRNA-seq and scATAC-seq training flows in a local environment with TF-IDF normalization and selectable loss functions.
              </p>
          </div>

          <div className="card p-6 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 rounded-lg bg-[rgb(var(--success-bg))] text-[rgb(var(--success-text))] flex items-center justify-center mb-4">
              <Zap className="w-6 h-6" />
            </div>
              <h3 className="text-lg font-semibold mb-2 text-[rgb(var(--foreground))]">Real-time Training</h3>
              <p className="text-[rgb(var(--text-secondary))]">
                Monitor training progress in a local session with live loss curves, epoch tracking, and checkpointing.
              </p>
          </div>

          <div className="card p-6 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 rounded-lg bg-[rgb(var(--training-bg))] text-[rgb(var(--training-text-bright))] flex items-center justify-center mb-4">
              <Eye className="w-6 h-6" />
            </div>
              <h3 className="text-lg font-semibold mb-2 text-[rgb(var(--foreground))]">Neural ODE Dynamics</h3>
              <p className="text-[rgb(var(--text-secondary))]">
                Inspect continuous cell trajectories with Neural ODE integration and extract latent plus interpretable embeddings for downstream analysis.
              </p>
          </div>
        </div>

        {/* Tech Stack */}
        <div className="mt-16 text-center">
          <p className="text-sm mb-4 text-[rgb(var(--muted-foreground))]">Built with</p>
          <div className="flex gap-6 justify-center items-center text-[rgb(var(--muted-foreground))]">
            <span className="font-medium">FastAPI</span>
            <span>•</span>
            <span className="font-medium">Next.js 15</span>
            <span>•</span>
            <span className="font-medium">PyTorch</span>
            <span>•</span>
            <span className="font-medium">Scanpy</span>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-20 border-t border-[rgb(var(--border))]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-center text-sm text-muted">
          <p>© 2025 iAODE Workspace. Local-first training surface with public-safe return paths to Homepage, SCPortal, and iAODE Pages.</p>
        </div>
      </footer>
    </div>
  );
}
