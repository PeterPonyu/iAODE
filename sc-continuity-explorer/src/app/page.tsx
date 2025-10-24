// ============================================================================
// FILE: app/page.tsx
// Home page with navigation
// ============================================================================

import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      {/* Hero Section */}
      <section className="text-center mb-16">
        <h1 className="text-4xl font-bold mb-4">
          Single-Cell Continuity Explorer
        </h1>
        <p className="text-lg text-[var(--color-muted-foreground)] mb-8">
          Interactive exploration of how continuity parameters affect trajectory structure
          in dimensionality reduction methods
        </p>
        <Link
          href="/explorer"
          className="inline-block px-6 py-3 bg-[var(--color-primary)] text-[var(--color-primary-foreground)] rounded-lg font-medium hover:opacity-90 transition-opacity"
        >
          Launch Explorer
        </Link>
      </section>

      {/* Features */}
      <section className="grid md:grid-cols-3 gap-8 mb-16">
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
      <section className="bg-[var(--color-muted)] rounded-lg p-8">
        <h2 className="text-2xl font-semibold mb-4">About This Tool</h2>
        <p className="text-[var(--color-muted-foreground)] leading-relaxed mb-4">
          This interactive tool allows researchers to explore pre-computed single-cell
          simulation data across varying continuity parameters and trajectory types.
        </p>
        <p className="text-[var(--color-muted-foreground)] leading-relaxed">
          All data is pre-computed and stored in chunked JSON files for fast loading
          and efficient exploration without requiring server-side computation.
        </p>
      </section>
    </div>
  );
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="p-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-background)]">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-sm text-[var(--color-muted-foreground)]">{description}</p>
    </div>
  );
}
