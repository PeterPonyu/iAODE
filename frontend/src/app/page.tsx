
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
            <span>AI-Powered Single-Cell Analysis</span>
          </div>

          <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-[rgb(var(--foreground))]">
            Interpretable Analysis of
            <br />
            <span className="bg-linear-to-r from-[rgb(var(--training-primary))] to-[rgb(var(--primary))] bg-clip-text text-transparent">
              Omics Data Explorer
            </span>
          </h1>
          
          <p className="text-xl max-w-2xl mx-auto text-[rgb(var(--text-secondary))]">
            Advanced single-cell RNA-seq and ATAC-seq analysis with interpretable
            embeddings powered by deep learning
          </p>

          <div className="flex gap-4 justify-center">
            <Link
              href="/train"
              className="btn-training px-8 py-3 rounded-lg font-medium"
            >
              Start Training
            </Link>
            <a
              href="https://github.com/PeterPonyu/iAODE"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary px-8 py-3 rounded-lg font-medium"
            >
              Documentation
            </a>
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
              Process scRNA-seq and scATAC-seq data with specialized loss functions (MSE, NB, ZINB)
            </p>
          </div>

          <div className="card p-6 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 rounded-lg bg-[rgb(var(--success-bg))] text-[rgb(var(--success-text))] flex items-center justify-center mb-4">
              <Zap className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold mb-2 text-[rgb(var(--foreground))]">Fast Training</h3>
            <p className="text-[rgb(var(--text-secondary))]">
              GPU-accelerated training with early stopping, validation monitoring, and batch processing
            </p>
          </div>

          <div className="card p-6 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 rounded-lg bg-[rgb(var(--training-bg))] text-[rgb(var(--training-text-bright))] flex items-center justify-center mb-4">
              <Eye className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold mb-2 text-[rgb(var(--foreground))]">Interpretable Results</h3>
            <p className="text-[rgb(var(--text-secondary))]">
              Extract both latent and interpretable embeddings for downstream analysis and visualization
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
          <p>© 2025 iAODE. Open source project for single-cell data analysis.</p>
        </div>
      </footer>
    </div>
  );
}
