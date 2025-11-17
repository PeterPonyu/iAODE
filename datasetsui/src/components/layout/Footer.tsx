import Link from 'next/link';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-[rgb(var(--border))] bg-[rgb(var(--background))] mt-auto">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="font-semibold text-[rgb(var(--foreground))] mb-3">
              About iAODE-VAE
            </h3>
            <p className="text-sm text-[rgb(var(--muted-foreground))] leading-relaxed">
              A curated collection of single-cell datasets supporting benchmarking research on 
              <strong className="text-[rgb(var(--foreground))]"> iAODE-VAE</strong> (Interpretable Accessibility ODE VAE). 
              All datasets are provided in standardized 10X h5 filtered matrix format, including peak matrices 
              for scATAC-seq and feature matrices for scRNA-seq, facilitating reproducible computational experiments 
              across multiple modalities.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-semibold text-[rgb(var(--foreground))] mb-3">Quick Links</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <Link 
                  href="/datasets" 
                  className="text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--primary-hover))] transition-colors"
                >
                  Browse Datasets
                </Link>
              </li>
              <li>
                <Link 
                  href="/statistics" 
                  className="text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--primary-hover))] transition-colors"
                >
                  Statistics
                </Link>
              </li>
              <li>
                <Link 
                  href="/about" 
                  className="text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--primary-hover))] transition-colors"
                >
                  About This Project
                </Link>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="font-semibold text-[rgb(var(--foreground))] mb-3">Data Resources</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a
                  href="https://www.ncbi.nlm.nih.gov/geo/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--primary-hover))] transition-colors inline-flex items-center"
                >
                  NCBI GEO Database
                  <svg className="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              </li>
              <li>
                <a
                  href="https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--primary-hover))] transition-colors inline-flex items-center"
                >
                  10X Genomics HDF5 Format
                  <svg className="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-[rgb(var(--border))] flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-[rgb(var(--muted-foreground))] text-center md:text-left">
            © {currentYear} iAODE-VAE Benchmarking Project. Datasets sourced from NCBI GEO and all in standardized 10X h5 format.
          </p>
          <p className="text-xs text-[rgb(var(--text-muted))] text-center md:text-right">
            Supporting computational experiments for deep learning models in single-cell omics.
          </p>
        </div>
      </div>
    </footer>
  );
}