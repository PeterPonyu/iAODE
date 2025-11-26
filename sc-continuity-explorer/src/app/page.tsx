
// ============================================================================
// FILE: app/page.tsx
// Home page with full-width layout
// ============================================================================

import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="w-full">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16 lg:py-20">
        {/* Hero Section */}
        <section className="text-center mb-16 sm:mb-20">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold mb-4 sm:mb-6 tracking-tight text-[rgb(var(--text-primary))]">
            iAODE Continuity Explorer
          </h1>
          <p className="text-base sm:text-lg text-[rgb(var(--muted-foreground))] mb-8 sm:mb-10 max-w-2xl mx-auto leading-relaxed">
            Interactive exploration of trajectory structures and continuity metrics
            across different embedding methods
          </p>
          <Link
            href="/explorer"
            className="btn-primary inline-block px-8 py-3.5 rounded-lg font-medium shadow-sm"
          >
            Launch Explorer
          </Link>
        </section>

        {/* Features */}
        <section className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-16 sm:mb-20">
          <FeatureCard
            title="Multiple Trajectories"
            description="Compare linear, branching, cyclic, and discrete cluster structures"
          />
          <FeatureCard
            title="Embedding Methods"
            description="Visualize with PCA, UMAP, and t-SNE dimensionality reduction"
          />
          <FeatureCard
            title="Continuity Analysis"
            description="Explore how continuity parameters affect structural preservation"
          />
        </section>

        {/* About */}
        <section className="bg-[rgb(var(--muted))] border border-[rgb(var(--border))] rounded-xl p-6 sm:p-8">
          <h2 className="text-2xl sm:text-3xl font-semibold mb-4 tracking-tight text-[rgb(var(--text-primary))]">
            About This Tool
          </h2>
          <div className="space-y-4 text-[rgb(var(--muted-foreground))] leading-relaxed">
            <p>
              This interactive tool allows researchers to explore pre-computed single-cell
              simulation data across varying continuity parameters and trajectory types.
            </p>
            <p>
              All data is pre-computed and stored in chunked JSON files for fast loading
              and efficient exploration without requiring server-side computation.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="card p-6 hover:shadow-md transition-shadow">
      <h3 className="text-lg font-semibold mb-2 tracking-tight text-[rgb(var(--text-primary))]">
        {title}
      </h3>
      <p className="text-sm text-[rgb(var(--muted-foreground))] leading-relaxed">
        {description}
      </p>
    </div>
  );
}
