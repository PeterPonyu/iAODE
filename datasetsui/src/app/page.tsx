
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
        <h1 className="text-4xl md:text-5xl font-bold text-[rgb(var(--foreground))] mb-4 transition-colors">
          scATAC-seq Dataset Browser
        </h1>
        <p className="text-xl text-[rgb(var(--muted-foreground))] max-w-2xl mx-auto transition-colors">
          Explore and download single-cell ATAC-seq datasets from NCBI GEO. 
          Browse {gseGroups.length} studies with {totalDatasets} datasets.
        </p>
      </section>

      {/* Quick Stats */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card text-center">
          <div className="text-4xl font-bold text-[rgb(var(--primary))] mb-2 transition-colors">
            {gseGroups.length}
          </div>
          <div className="text-[rgb(var(--muted-foreground))] transition-colors">GSE Studies</div>
        </div>
        
        <div className="card text-center">
          <div className="text-4xl font-bold text-[rgb(var(--primary))] mb-2 transition-colors">
            {totalDatasets}
          </div>
          <div className="text-[rgb(var(--muted-foreground))] transition-colors">Total Datasets</div>
        </div>
        
        <div className="card text-center">
          <div className="text-4xl font-bold text-[rgb(var(--primary))] mb-2 transition-colors">
            {(totalCells / 1000000).toFixed(1)}M
          </div>
          <div className="text-[rgb(var(--muted-foreground))] transition-colors">Total Cells</div>
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
