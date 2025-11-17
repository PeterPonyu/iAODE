import Link from 'next/link';
import { GSEGroup } from '@/types/datasets';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { ExternalLink } from 'lucide-react';

interface DatasetTableProps {
  data: GSEGroup[];
  dataType: 'ATAC' | 'RNA';
}

export default function DatasetTable({ data, dataType }: DatasetTableProps) {
  const featureLabel = dataType === 'ATAC' ? 'Peaks' : 'Genes';
  const featureColor = dataType === 'ATAC' 
    ? 'text-[rgb(var(--atac-primary))]' 
    : 'text-[rgb(var(--rna-primary))]';

  return (
    <div className="card p-0 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-[rgb(var(--secondary))] border-b border-[rgb(var(--border-light))] transition-colors">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))] transition-colors">
                GSE ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))] transition-colors">
                Title
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))] transition-colors">
                Organism
              </th>
              <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))] transition-colors">
                Datasets
              </th>
              <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))] transition-colors">
                Cells
              </th>
              <th className={`px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider transition-colors ${featureColor}`}>
                {featureLabel}
              </th>
              <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))] transition-colors">
                Size
              </th>
              <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))] transition-colors">
                
              </th>
            </tr>
          </thead>
          <tbody className="bg-[rgb(var(--background))] transition-colors">
            {data.map((gseGroup) => (
              <tr
                key={gseGroup.gseAccession}
                className="border-b border-[rgb(var(--border-light))] transition-colors hover:bg-[rgb(var(--muted))]"
              >
                <td className="px-6 py-4 whitespace-nowrap text-[rgb(var(--foreground))] transition-colors">
                  <Link
                    href={`/datasets/${gseGroup.gseAccession}?type=${dataType}`}
                    className={`text-sm font-medium transition-colors ${
                      dataType === 'ATAC'
                        ? 'text-[rgb(var(--atac-primary))] hover:text-[rgb(var(--atac-primary-hover))]'
                        : 'text-[rgb(var(--rna-primary))] hover:text-[rgb(var(--rna-primary-hover))]'
                    }`}
                  >
                    {gseGroup.gseAccession}
                  </Link>
                </td>
                <td className="px-6 py-4 text-[rgb(var(--foreground))] transition-colors">
                  <div className="text-sm font-medium max-w-md truncate">
                    {gseGroup.title}
                  </div>
                  <div className="text-xs text-[rgb(var(--muted-foreground))] transition-colors truncate">
                    {gseGroup.authors}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-[rgb(var(--foreground))] transition-colors">
                  {gseGroup.organism}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-[rgb(var(--foreground))] transition-colors">
                  {gseGroup.datasets.length}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-[rgb(var(--foreground))] transition-colors">
                  {formatNumber(gseGroup.totalCells, true)}
                </td>
                <td className={`px-6 py-4 whitespace-nowrap text-right text-sm font-medium transition-colors ${featureColor}`}>
                  {formatNumber(gseGroup.totalFeatures, true)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-[rgb(var(--foreground))] transition-colors">
                  {formatFileSize(gseGroup.totalSize)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                  <Link
                    href={`/datasets/${gseGroup.gseAccession}?type=${dataType}`}
                    className={`transition-colors inline-block ${
                      dataType === 'ATAC'
                        ? 'text-[rgb(var(--atac-primary))] hover:text-[rgb(var(--atac-primary-hover))]'
                        : 'text-[rgb(var(--rna-primary))] hover:text-[rgb(var(--rna-primary-hover))]'
                    }`}
                    aria-label="View details"
                  >
                    <ExternalLink className="h-4 w-4" />
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}