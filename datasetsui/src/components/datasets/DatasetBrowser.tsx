'use client';

import { useState, useMemo } from 'react';
import { GSEGroup, FilterState } from '@/types/datasets';
import DatasetFilters from './DatasetFilters';
import DatasetSearch from './DatasetSearch';
import DatasetGrid from './DatasetGrid';
import DatasetTable from './DatasetTable';
import ViewToggle from './ViewToggle';
import CategoryLegend from '@/components/ui/CategoryLegend';
import { applyFilters } from '@/lib/filterUtils';
import { searchGSE } from '@/lib/searchUtils';

interface DatasetBrowserProps {
  initialData: GSEGroup[];
}

export default function DatasetBrowser({ initialData }: DatasetBrowserProps) {
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    categories: [],
    organisms: [],
    cellRange: null,
  });

  // Apply search first (on GSE level)
  const searchedGSEGroups = useMemo(() => 
    searchGSE(initialData, searchQuery),
    [initialData, searchQuery]
  );

  // Then apply filters (on dataset level within each GSE)
  const filteredGSEGroups = useMemo(() => {
    return searchedGSEGroups
      .map(gse => ({
        ...gse,
        datasets: applyFilters(gse.datasets, filters)
      }))
      .filter(gse => gse.datasets.length > 0);
  }, [searchedGSEGroups, filters]);

  const totalDatasets = filteredGSEGroups.reduce(
    (sum, g) => sum + g.datasets.length, 
    0
  );

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      {/* Filters Sidebar */}
      <aside className="lg:w-64 flex-shrink-0">
        <div className="lg:sticky lg:top-4 space-y-4">
          <DatasetFilters
            filters={filters}
            onFiltersChange={setFilters}
            allDatasets={initialData.flatMap(g => g.datasets)}
          />
          {/* Legend in sidebar for desktop */}
          <div className="hidden lg:block">
            <CategoryLegend compact={false} />
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 min-w-0">
        {/* Search & Controls */}
        <div className="space-y-4 mb-6">
          <DatasetSearch
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Search by GSE ID, title, authors, or organism..."
          />
          
          {/* Compact Legend for mobile/tablet */}
          <div className="lg:hidden">
            <CategoryLegend compact showTitle />
          </div>
          
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Showing <span className="font-semibold text-gray-900 dark:text-gray-100">{filteredGSEGroups.length}</span> studies 
              ({totalDatasets} datasets) of {initialData.length} total
            </p>
            <ViewToggle mode={viewMode} onChange={setViewMode} />
          </div>
        </div>

        {/* Results */}
        {filteredGSEGroups.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-gray-400 dark:text-gray-500 mb-4">
              <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
              No datasets found
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Try adjusting your search or filters
            </p>
            <button
              onClick={() => {
                setSearchQuery('');
                setFilters({
                  search: '',
                  categories: [],
                  organisms: [],
                  cellRange: null,
                });
              }}
              className="btn-secondary"
            >
              Clear all filters
            </button>
          </div>
        ) : viewMode === 'grid' ? (
          <DatasetGrid data={filteredGSEGroups} />
        ) : (
          <DatasetTable data={filteredGSEGroups} />
        )}
      </div>
    </div>
  );
}