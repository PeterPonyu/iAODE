'use client';

import { FilterState, MergedDataset } from '@/types/datasets';
import { getFilterOptions } from '@/lib/filterUtils';
import { useMemo } from 'react';
import { X } from 'lucide-react';

interface DatasetFiltersProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  allDatasets: MergedDataset[];
}

export default function DatasetFilters({
  filters,
  onFiltersChange,
  allDatasets
}: DatasetFiltersProps) {
  const options = useMemo(() => getFilterOptions(allDatasets), [allDatasets]);

  const toggleCategory = (cat: FilterState['categories'][number]) => {
    const newCategories = filters.categories.includes(cat)
      ? filters.categories.filter(c => c !== cat)
      : [...filters.categories, cat];
    onFiltersChange({ ...filters, categories: newCategories });
  };

  const toggleOrganism = (org: string) => {
    const newOrganisms = filters.organisms.includes(org)
      ? filters.organisms.filter(o => o !== org)
      : [...filters.organisms, org];
    onFiltersChange({ ...filters, organisms: newOrganisms });
  };

  const resetFilters = () => {
    onFiltersChange({
      search: '',
      categories: [],
      organisms: [],
      cellRange: null,
    });
  };

  const activeFilterCount = 
    filters.categories.length + 
    filters.organisms.length + 
    (filters.cellRange ? 1 : 0);

  return (
    <div className="card p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-lg text-[rgb(var(--foreground))]">
          Filters
        </h2>
        {activeFilterCount > 0 && (
          <button
            onClick={resetFilters}
            className="text-sm text-[rgb(var(--primary))] transition-colors hover:text-[rgb(var(--primary-hover))] flex items-center gap-1 font-medium"
          >
            <X className="h-4 w-4" />
            Reset ({activeFilterCount})
          </button>
        )}
      </div>

      {/* Category Filter */}
      <div>
        <h3 className="font-semibold text-sm text-[rgb(var(--foreground))] mb-3">
          🔖 Dataset Size
        </h3>
        <div className="space-y-2">
          {options.categories.map(cat => {
            const count = allDatasets.filter(d => d.category === cat).length;
            return (
              <label
                key={cat}
                className="flex items-center gap-2 cursor-pointer group"
              >
                <input
                  type="checkbox"
                  checked={filters.categories.includes(cat)}
                  onChange={() => toggleCategory(cat)}
                  className="rounded border-[rgb(var(--border))] cursor-pointer transition-all checked:bg-[rgb(var(--primary))] checked:border-[rgb(var(--primary))] focus:outline-none focus:ring-3 focus:ring-[rgb(var(--primary))] focus:ring-opacity-20"
                />
                <span className="text-sm text-[rgb(var(--text-secondary))] transition-colors group-hover:text-[rgb(var(--muted-foreground))] capitalize">
                  {cat}
                </span>
                <span className="text-xs text-[rgb(var(--muted-foreground))] ml-auto font-medium">
                  {count}
                </span>
              </label>
            );
          })}
        </div>
      </div>

      {/* Organism Filter */}
      <div>
        <h3 className="font-semibold text-sm text-[rgb(var(--foreground))] mb-3">
          🧬 Organism
        </h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {options.organisms.map(org => {
            const count = allDatasets.filter(d => d.organism === org).length;
            return (
              <label
                key={org}
                className="flex items-center gap-2 cursor-pointer group"
              >
                <input
                  type="checkbox"
                  checked={filters.organisms.includes(org)}
                  onChange={() => toggleOrganism(org)}
                  className="rounded border-[rgb(var(--border))] cursor-pointer transition-all checked:bg-[rgb(var(--primary))] checked:border-[rgb(var(--primary))] focus:outline-none focus:ring-3 focus:ring-[rgb(var(--primary))] focus:ring-opacity-20"
                />
                <span className="text-sm text-[rgb(var(--text-secondary))] transition-colors group-hover:text-[rgb(var(--muted-foreground))]">
                  {org}
                </span>
                <span className="text-xs text-[rgb(var(--muted-foreground))] ml-auto font-medium">
                  {count}
                </span>
              </label>
            );
          })}
        </div>
      </div>
    </div>
  );
}