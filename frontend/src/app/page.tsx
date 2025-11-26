
'use client';

import Link from 'next/link';
import { Header } from '@/components/header';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      {/* Hero Section */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center space-y-8">
          <h2 className="text-5xl font-bold tracking-tight text-[rgb(var(--text-primary))]">
            Interpretable Analysis of
            <br />
            <span className="text-[rgb(var(--primary))]">Omics Data Explorer</span>
          </h2>
          
          <p className="text-xl max-w-2xl mx-auto text-[rgb(var(--muted-foreground))]">
            Advanced single-cell RNA-seq and ATAC-seq analysis with interpretable
            embeddings powered by deep learning
          </p>

          <div className="flex gap-4 justify-center">
            <Link
              href="/train"
              className="btn-primary px-8 py-3 rounded-lg font-medium"
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
          <div className="card">
            <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 badge-blue">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-[rgb(var(--text-primary))]">Multi-modal Support</h3>
            <p className="text-[rgb(var(--muted-foreground))]">
              Process scRNA-seq and scATAC-seq data with specialized loss functions (MSE, NB, ZINB)
            </p>
          </div>

          <div className="card">
            <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 badge-green">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-[rgb(var(--text-primary))]">Fast Training</h3>
            <p className="text-[rgb(var(--muted-foreground))]">
              GPU-accelerated training with early stopping, validation monitoring, and batch processing
            </p>
          </div>

          <div className="card">
            <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 badge-purple">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-[rgb(var(--text-primary))]">Interpretable Results</h3>
            <p className="text-[rgb(var(--muted-foreground))]">
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
      <footer className="border-t border-[rgb(var(--border))] mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-center text-sm text-[rgb(var(--muted-foreground))]">
          <p>© 2025 iAODE. Open source project for single-cell data analysis.</p>
        </div>
      </footer>
    </div>
  );
}
