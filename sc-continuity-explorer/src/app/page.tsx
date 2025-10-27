
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
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold mb-4 sm:mb-6 tracking-tight">
            Single-Cell Continuity Explorer
          </h1>
          <p className="text-base sm:text-lg text-[var(--color-muted-foreground)] mb-8 sm:mb-10 max-w-2xl mx-auto leading-relaxed">
            Interactive exploration of how continuity parameters affect trajectory structure
            in dimensionality reduction methods
          </p>
          <Link
            href="/explorer"
            className="inline-block px-8 py-3.5 bg-[var(--color-primary)] text-[var(--color-primary-foreground)] rounded-lg font-medium hover:opacity-90 transition-opacity shadow-sm"
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
        <section className="bg-[var(--color-muted)] rounded-xl p-6 sm:p-8">
          <h2 className="text-2xl sm:text-3xl font-semibold mb-4 tracking-tight">
            About This Tool
          </h2>
          <div className="space-y-4 text-[var(--color-muted-foreground)] leading-relaxed">
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
    <div className="p-6 rounded-xl border border-[var(--color-border)] bg-[var(--color-background)] hover:shadow-md transition-shadow">
      <h3 className="text-lg font-semibold mb-2 tracking-tight">
        {title}
      </h3>
      <p className="text-sm text-[var(--color-muted-foreground)] leading-relaxed">
        {description}
      </p>
    </div>
  );
}
