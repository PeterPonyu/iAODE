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
              scATAC-seq Dataset Browser
            </h3>
            <p className="text-sm text-[rgb(var(--muted-foreground))]">
              A comprehensive collection of single-cell ATAC-seq datasets for research and analysis.
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
                  About
                </Link>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="font-semibold text-[rgb(var(--foreground))] mb-3">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a
                  href="https://www.ncbi.nlm.nih.gov/geo/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--primary-hover))] transition-colors inline-flex items-center"
                >
                  NCBI GEO
                  <svg className="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-[rgb(var(--border))] flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-[rgb(var(--muted-foreground))]">
            © {currentYear} scATAC-seq Dataset Browser. All rights reserved.
          </p>
          <div className="flex space-x-4 mt-4 md:mt-0">
            {/* Add your social/contact links here */}
          </div>
        </div>
      </div>
    </footer>
  );
}