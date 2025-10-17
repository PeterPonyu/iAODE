// app/page.tsx

import { getAllGSEGroups } from '@/lib/dataLoader';

export default function HomePage() {
  const gseGroups = getAllGSEGroups();
  const totalDatasets = gseGroups.reduce((sum, g) => sum + g.datasets.length, 0);
  const totalCells = gseGroups.reduce((sum, g) => sum + g.totalCells, 0);

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center py-12">
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          scATAC-seq Dataset Browser
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          Explore and download single-cell ATAC-seq datasets from NCBI GEO. 
          Browse {gseGroups.length} studies with {totalDatasets} datasets.
        </p>
      </section>

      {/* Quick Stats */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card text-center">
          <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
            {gseGroups.length}
          </div>
          <div className="text-gray-600 dark:text-gray-400">GSE Studies</div>
        </div>
        
        <div className="card text-center">
          <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
            {totalDatasets}
          </div>
          <div className="text-gray-600 dark:text-gray-400">Total Datasets</div>
        </div>
        
        <div className="card text-center">
          <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
            {(totalCells / 1000000).toFixed(1)}M
          </div>
          <div className="text-gray-600 dark:text-gray-400">Total Cells</div>
        </div>
      </section>

      {/* CTA */}
      <section className="text-center">
        <a href="/datasets" className="btn-primary text-lg">
          Browse All Datasets →
        </a>
      </section>
    </div>
  );
}