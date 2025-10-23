import { GSEGroup } from '@/types/datasets';
import { generateGeoUrl } from '@/lib/geoUtils';
import { ExternalLink, User, Dna, Cpu } from 'lucide-react';

interface GSEHeaderProps {
  gseGroup: GSEGroup;
}

export default function GSEHeader({ gseGroup }: GSEHeaderProps) {
  const { gseAccession, title, authors, organism, platforms } = gseGroup;

  return (
    <div className="card p-6">
      {/* Title Row */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-4">
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-[rgb(var(--foreground))] mb-2">
            {gseAccession}
          </h1>
          <p className="text-lg text-[rgb(var(--text-secondary))]">
            {title}
          </p>
        </div>

        {/* External Link Button */}
        <a
          href={generateGeoUrl(gseAccession)}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary inline-flex items-center gap-2 whitespace-nowrap"
        >
          <ExternalLink className="h-4 w-4" />
          View on NCBI GEO
        </a>
      </div>

      {/* Metadata Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-[rgb(var(--border-light))]">
        {/* Authors */}
        <div className="flex items-start gap-3">
          <div className="mt-1">
            <User className="h-5 w-5 text-[rgb(var(--text-muted))]" />
          </div>
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] uppercase tracking-wide mb-1">
              Authors
            </p>
            <p className="text-sm text-[rgb(var(--foreground))]">
              {authors}
            </p>
          </div>
        </div>

        {/* Organism */}
        <div className="flex items-start gap-3">
          <div className="mt-1">
            <Dna className="h-5 w-5 text-[rgb(var(--text-muted))]" />
          </div>
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] uppercase tracking-wide mb-1">
              Organism
            </p>
            <p className="text-sm text-[rgb(var(--foreground))]">
              {organism}
            </p>
          </div>
        </div>

        {/* Platforms */}
        <div className="flex items-start gap-3">
          <div className="mt-1">
            <Cpu className="h-5 w-5 text-[rgb(var(--text-muted))]" />
          </div>
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] uppercase tracking-wide mb-1">
              Platform{platforms.length !== 1 ? 's' : ''}
            </p>
            <p className="text-sm text-[rgb(var(--foreground))]">
              {platforms.length > 0 ? platforms.join(', ') : 'Not specified'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
