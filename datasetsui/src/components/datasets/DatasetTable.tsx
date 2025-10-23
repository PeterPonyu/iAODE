import Link from 'next/link';
import { GSEGroup } from '@/types/datasets';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { ExternalLink } from 'lucide-react';

interface DatasetTableProps {
  data: GSEGroup[];
}

export default function DatasetTable({ data }: DatasetTableProps) {
  return (
    <div className="card p-0 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-[rgb(var(--secondary))] border-b border-[rgb(var(--border-light))]">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))]">
                GSE ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))]">
                Title
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))]">
                Organism
              </th>
              <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))]">
                Datasets
              </th>
              <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))]">
                Cells
              </th>
              <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))]">
                Size
              </th>
              <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-[rgb(var(--text-secondary))]">
                
              </th>
            </tr>
          </thead>
          <tbody className="bg-[rgb(var(--background))]">
            {data.map((gseGroup) => (
              <tr
                key={gseGroup.gseAccession}
                className="border-b border-[rgb(var(--border-light))] transition-colors hover:bg-[rgb(var(--muted))]"
              >
                <td className="px-6 py-4 whitespace-nowrap text-[rgb(var(--foreground))]">
                  <Link
                    href={`/datasets/${gseGroup.gseAccession}`}
                    className="text-sm font-medium text-[rgb(var(--primary))] transition-colors hover:text-[rgb(var(--primary-hover))]"
                  >
                    {gseGroup.gseAccession}
                  </Link>
                </td>
                <td className="px-6 py-4 text-[rgb(var(--foreground))]">
                  <div className="text-sm font-medium max-w-md truncate">
                    {gseGroup.title}
                  </div>
                  <div className="text-xs text-[rgb(var(--muted-foreground))] truncate">
                    {gseGroup.authors}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-[rgb(var(--foreground))]">
                  {gseGroup.organism}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-[rgb(var(--foreground))]">
                  {gseGroup.datasets.length}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-[rgb(var(--foreground))]">
                  {formatNumber(gseGroup.totalCells, true)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-[rgb(var(--foreground))]">
                  {formatFileSize(gseGroup.totalSize)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                  <Link
                    href={`/datasets/${gseGroup.gseAccession}`}
                    className="text-[rgb(var(--primary))] transition-colors hover:text-[rgb(var(--primary-hover))] inline-block"
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