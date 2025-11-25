// app/page.tsx
// 'use client';

import { getAllGSEGroups } from '@/lib/dataLoader';
import { Activity, Dna } from 'lucide-react';
import Link from 'next/link';

export default function HomePage() {
  const gseGroupsATAC = getAllGSEGroups('ATAC');
  const totalDatasetsATAC = gseGroupsATAC.reduce((sum, g) => sum + g.datasets.length, 0);
  const totalCellsATAC = gseGroupsATAC.reduce((sum, g) => sum + g.totalCells, 0);
  const totalFeaturesATAC = gseGroupsATAC.reduce((sum, g) => sum + g.totalFeatures, 0);

  const gseGroupsRNA = getAllGSEGroups('RNA');
  const totalDatasetsRNA = gseGroupsRNA.reduce((sum, g) => sum + g.datasets.length, 0);
  const totalCellsRNA = gseGroupsRNA.reduce((sum, g) => sum + g.totalCells, 0);
  const totalFeaturesRNA = gseGroupsRNA.reduce((sum, g) => sum + g.totalFeatures, 0);

  return (
    <div className="space-y-16">
      {/* Header */}
      <section className="text-center py-12">
        <h1 className="text-5xl md:text-6xl font-bold text-[rgb(var(--text-primary))] mb-4 transition-colors">
          iAODE Dataset Browser
        </h1>
        <p className="text-xl text-[rgb(var(--text-secondary))] max-w-2xl mx-auto transition-colors">
          Explore comprehensive single-cell ATAC-seq and RNA-seq datasets from NCBI GEO
        </p>
      </section>

      {/* Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
        
        {/* scATAC Section */}
        <div className="space-y-8">
          {/* scATAC Header with Icon */}
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 bg-[rgb(var(--atac-icon-bg))] rounded-lg transition-colors">
              <Activity className="w-8 h-8 text-[rgb(var(--atac-icon))] transition-colors" />
            </div>
            <div>
              <h2 className="text-3xl font-bold text-[rgb(var(--atac-text))] transition-colors">
                scATAC-seq
              </h2>
              <p className="text-sm text-[rgb(var(--atac-text-bright))] transition-colors">
                Chromatin Accessibility
              </p>
            </div>
          </div>

          {/* scATAC Description */}
          <div className="bg-[rgb(var(--atac-bg-subtle))] border border-[rgb(var(--atac-border))] rounded-lg p-4 transition-colors">
            <p className="text-[rgb(var(--atac-text))] text-sm transition-colors">
              Explore {gseGroupsATAC.length} GSE studies with {totalDatasetsATAC} ATAC-seq datasets.
              Analyze chromatin peaks across {totalCellsATAC.toLocaleString()} cells.
            </p>
          </div>

          {/* scATAC Stats */}
          <section className="grid grid-cols-3 gap-3">
            <div className="card-atac text-center">
              <div className="text-2xl font-bold text-[rgb(var(--atac-primary))] mb-1 transition-colors">
                {gseGroupsATAC.length}
              </div>
              <div className="text-xs text-[rgb(var(--text-tertiary))] transition-colors">
                GSE Studies
              </div>
            </div>
            
            <div className="card-atac text-center">
              <div className="text-2xl font-bold text-[rgb(var(--atac-primary))] mb-1 transition-colors">
                {totalDatasetsATAC}
              </div>
              <div className="text-xs text-[rgb(var(--text-tertiary))] transition-colors">
                Datasets
              </div>
            </div>
            
            <div className="card-atac text-center">
              <div className="text-2xl font-bold text-[rgb(var(--atac-primary))] mb-1 transition-colors">
                {(totalFeaturesATAC / 1000000).toFixed(1)}M
              </div>
              <div className="text-xs text-[rgb(var(--text-tertiary))] transition-colors">
                Peaks
              </div>
            </div>
          </section>

          {/* scATAC CTA */}
          <Link href="/datasets?type=ATAC" className="btn-atac w-full block text-center">
            Browse scATAC Datasets →
          </Link>
        </div>

        {/* scRNA Section */}
        <div className="space-y-8">
          {/* scRNA Header with Icon */}
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 bg-[rgb(var(--rna-icon-bg))] rounded-lg transition-colors">
              <Dna className="w-8 h-8 text-[rgb(var(--rna-icon))] transition-colors" />
            </div>
            <div>
              <h2 className="text-3xl font-bold text-[rgb(var(--rna-text))] transition-colors">
                scRNA-seq
              </h2>
              <p className="text-sm text-[rgb(var(--rna-text-bright))] transition-colors">
                Gene Expression
              </p>
            </div>
          </div>

          {/* scRNA Description */}
          <div className="bg-[rgb(var(--rna-bg-subtle))] border border-[rgb(var(--rna-border))] rounded-lg p-4 transition-colors">
            <p className="text-[rgb(var(--rna-text))] text-sm transition-colors">
              Explore {gseGroupsRNA.length} GSE studies with {totalDatasetsRNA} RNA-seq datasets.
              Analyze gene expression across {totalCellsRNA.toLocaleString()} cells.
            </p>
          </div>

          {/* scRNA Stats */}
          <section className="grid grid-cols-3 gap-3">
            <div className="card-rna text-center">
              <div className="text-2xl font-bold text-[rgb(var(--rna-primary))] mb-1 transition-colors">
                {gseGroupsRNA.length}
              </div>
              <div className="text-xs text-[rgb(var(--text-tertiary))] transition-colors">
                GSE Studies
              </div>
            </div>
            
            <div className="card-rna text-center">
              <div className="text-2xl font-bold text-[rgb(var(--rna-primary))] mb-1 transition-colors">
                {totalDatasetsRNA}
              </div>
              <div className="text-xs text-[rgb(var(--text-tertiary))] transition-colors">
                Datasets
              </div>
            </div>
            
            <div className="card-rna text-center">
              <div className="text-2xl font-bold text-[rgb(var(--rna-primary))] mb-1 transition-colors">
                {(totalFeaturesRNA / 1000000).toFixed(1)}M
              </div>
              <div className="text-xs text-[rgb(var(--text-tertiary))] transition-colors">
                Genes
              </div>
            </div>
          </section>

          {/* scRNA CTA */}
          <Link href="/datasets?type=RNA" className="btn-rna w-full block text-center">
            Browse scRNA Datasets →
          </Link>
        </div>
      </div>

      {/* Footer Stats */}
      <section className="border-t border-[rgb(var(--border))] pt-12 mt-8 transition-colors">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          <div>
            <div className="text-3xl font-bold text-[rgb(var(--text-primary))] mb-1 transition-colors">
              {gseGroupsATAC.length + gseGroupsRNA.length}
            </div>
            <div className="text-sm text-[rgb(var(--text-tertiary))] transition-colors">
              Total GSE Studies
            </div>
          </div>
          
          <div>
            <div className="text-3xl font-bold text-[rgb(var(--text-primary))] mb-1 transition-colors">
              {totalDatasetsATAC + totalDatasetsRNA}
            </div>
            <div className="text-sm text-[rgb(var(--text-tertiary))] transition-colors">
              Total Datasets
            </div>
          </div>
          
          <div>
            <div className="text-3xl font-bold text-[rgb(var(--text-primary))] mb-1 transition-colors">
              {((totalCellsATAC + totalCellsRNA) / 1000000).toFixed(1)}M
            </div>
            <div className="text-sm text-[rgb(var(--text-tertiary))] transition-colors">
              Total Cells
            </div>
          </div>
          
          <div>
            <div className="text-3xl font-bold text-[rgb(var(--text-primary))] mb-1 transition-colors">
              2
            </div>
            <div className="text-sm text-[rgb(var(--text-tertiary))] transition-colors">
              Omics Types
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}