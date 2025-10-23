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
        <h2 className="font-semibold text-lg text-gray-900 dark:text-gray-100">
          Filters
        </h2>
        {activeFilterCount > 0 && (
          <button
            onClick={resetFilters}
            className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 flex items-center gap-1"
          >
            <X className="h-4 w-4" />
            Reset ({activeFilterCount})
          </button>
        )}
      </div>

      {/* Category Filter */}
      <div>
        <h3 className="font-medium text-sm text-gray-700 dark:text-gray-300 mb-3">
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
                  className="
                    rounded border-gray-300 dark:border-gray-600
                    text-blue-600 focus:ring-blue-500
                    cursor-pointer
                  "
                />
                <span className="text-sm text-gray-700 dark:text-gray-300 capitalize group-hover:text-gray-900 dark:group-hover:text-gray-100">
                  {cat}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-auto">
                  {count}
                </span>
              </label>
            );
          })}
        </div>
      </div>

      {/* Organism Filter */}
      <div>
        <h3 className="font-medium text-sm text-gray-700 dark:text-gray-300 mb-3">
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
                  className="
                    rounded border-gray-300 dark:border-gray-600
                    text-blue-600 focus:ring-blue-500
                    cursor-pointer
                  "
                />
                <span className="text-sm text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-gray-100">
                  {org}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-auto">
                  {count}
                </span>
              </label>
            );
          })}
        </div>
      </div>

      {/* Cell Range Filter - Can add later with a slider component */}
      {/* <div>
        <h3 className="font-medium text-sm text-gray-700 dark:text-gray-300 mb-3">
          🔢 Cell Count Range
        </h3>
        <RangeSlider 
          min={options.cellRange.min}
          max={options.cellRange.max}
          value={filters.cellRange}
          onChange={(range) => onFiltersChange({ ...filters, cellRange: range })}
        />
      </div> */}
    </div>
  );
}