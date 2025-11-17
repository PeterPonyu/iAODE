import { GSEGroup } from '@/types/datasets';
import { generateGeoUrl } from '@/lib/geoUtils';
import { ExternalLink, User, Dna, Cpu, Activity } from 'lucide-react';

interface GSEHeaderProps {
  gseGroup: GSEGroup;
  dataType: 'ATAC' | 'RNA';
}

export default function GSEHeader({ gseGroup, dataType }: GSEHeaderProps) {
  const { gseAccession, title, authors, organism, platforms } = gseGroup;
  
  const DataTypeIcon = dataType === 'ATAC' ? Activity : Dna;
  const dataTypeLabel = dataType === 'ATAC' ? 'scATAC-seq' : 'scRNA-seq';
  const dataTypeColor = dataType === 'ATAC'
    ? 'text-[rgb(var(--atac-primary))] bg-[rgb(var(--atac-bg-subtle))] border-[rgb(var(--atac-border))]'
    : 'text-[rgb(var(--rna-primary))] bg-[rgb(var(--rna-bg-subtle))] border-[rgb(var(--rna-border))]';

  return (
    <div className="card p-6">
      {/* Data Type Badge */}
      <div className="mb-4">
        <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm font-medium transition-colors ${dataTypeColor}`}>
          <DataTypeIcon className="h-4 w-4" />
          {dataTypeLabel} Study
        </span>
      </div>

      {/* Title Row */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-4">
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-[rgb(var(--foreground))] transition-colors mb-2">
            {gseAccession}
          </h1>
          <p className="text-lg text-[rgb(var(--text-secondary))] transition-colors">
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
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-[rgb(var(--border-light))] transition-colors">
        {/* Authors */}
        <div className="flex items-start gap-3">
          <div className="mt-1">
            <User className="h-5 w-5 text-[rgb(var(--text-muted))] transition-colors" />
          </div>
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] uppercase tracking-wide mb-1 transition-colors">
              Authors
            </p>
            <p className="text-sm text-[rgb(var(--foreground))] transition-colors">
              {authors}
            </p>
          </div>
        </div>

        {/* Organism */}
        <div className="flex items-start gap-3">
          <div className="mt-1">
            <Dna className="h-5 w-5 text-[rgb(var(--text-muted))] transition-colors" />
          </div>
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] uppercase tracking-wide mb-1 transition-colors">
              Organism
            </p>
            <p className="text-sm text-[rgb(var(--foreground))] transition-colors">
              {organism}
            </p>
          </div>
        </div>

        {/* Platforms */}
        <div className="flex items-start gap-3">
          <div className="mt-1">
            <Cpu className="h-5 w-5 text-[rgb(var(--text-muted))] transition-colors" />
          </div>
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] uppercase tracking-wide mb-1 transition-colors">
              Platform{platforms.length !== 1 ? 's' : ''}
            </p>
            <p className="text-sm text-[rgb(var(--foreground))] transition-colors">
              {platforms.length > 0 ? platforms.join(', ') : 'Not specified'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}