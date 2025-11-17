'use client';

import { useState, useMemo, useEffect } from 'react';
import { getAllGSEGroups } from '@/lib/dataLoader';
import { calculateStats } from '@/lib/statsCalculator';
import StatisticsOverview from '@/components/statistics/StatisticsOverview';
import CategoryDistribution from '@/components/statistics/CategoryDistribution';
import OrganismDistribution from '@/components/statistics/OrganismDistribution';
import PlatformDistribution from '@/components/statistics/PlatformDistribution';
import CellsDistribution from '@/components/statistics/CellsDistribution';
import FeaturesDistribution from '@/components/statistics/FeaturesDistribution';
import ComparisonCard from '@/components/statistics/ComparisonCard';
import { useSearchParams, useRouter } from 'next/navigation';
import { Activity, Dna } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function StatisticsPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialType = (searchParams.get('type') as 'ATAC' | 'RNA') || 'ATAC';
  const [dataType, setDataType] = useState<'ATAC' | 'RNA'>(initialType);
  const [showComparison, setShowComparison] = useState(false);

  // Sync state with URL on mount/change
  useEffect(() => {
    const urlType = searchParams.get('type') as 'ATAC' | 'RNA';
    if (urlType && urlType !== dataType) {
      setDataType(urlType);
    }
  }, [searchParams]);

  // Load data for both types
  const gseGroupsATAC = useMemo(() => getAllGSEGroups('ATAC'), []);
  const gseGroupsRNA = useMemo(() => getAllGSEGroups('RNA'), []);

  const statsATAC = useMemo(() => calculateStats(gseGroupsATAC, 'ATAC'), [gseGroupsATAC]);
  const statsRNA = useMemo(() => calculateStats(gseGroupsRNA, 'RNA'), [gseGroupsRNA]);

  // Current data based on selected type
  const currentGSEGroups = dataType === 'ATAC' ? gseGroupsATAC : gseGroupsRNA;
  const currentStats = dataType === 'ATAC' ? statsATAC : statsRNA;

  const handleTypeChange = (type: 'ATAC' | 'RNA') => {
    setDataType(type);
    router.push(`/statistics?type=${type}`, { scroll: false });
  };

  const featureLabel = dataType === 'ATAC' ? 'peaks' : 'genes';

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-[rgb(var(--foreground))] mb-2 transition-colors">
          Dataset Statistics
        </h1>
        <p className="text-[rgb(var(--muted-foreground))] transition-colors">
          Comprehensive analysis and visualizations of single-cell genomics datasets
        </p>
      </div>

      {/* Data Type Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="inline-flex rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--card))] p-1 transition-colors">
          <button
            onClick={() => handleTypeChange('ATAC')}
            className={cn(
              'inline-flex items-center gap-2 px-4 py-2.5 rounded-md text-sm font-medium transition-all',
              dataType === 'ATAC'
                ? 'bg-[rgb(var(--atac-primary))] text-white shadow-sm'
                : 'text-[rgb(var(--text-secondary))] hover:bg-[rgb(var(--muted))]'
            )}
          >
            <Activity className="h-4 w-4" />
            <span>scATAC-seq</span>
            <span className={cn(
              "ml-1 px-2 py-0.5 rounded-full text-xs",
              dataType === 'ATAC' ? 'bg-white/20' : 'bg-[rgb(var(--muted))]'
            )}>
              {statsATAC.totalDatasets}
            </span>
          </button>
          <button
            onClick={() => handleTypeChange('RNA')}
            className={cn(
              'inline-flex items-center gap-2 px-4 py-2.5 rounded-md text-sm font-medium transition-all',
              dataType === 'RNA'
                ? 'bg-[rgb(var(--rna-primary))] text-white shadow-sm'
                : 'text-[rgb(var(--text-secondary))] hover:bg-[rgb(var(--muted))]'
            )}
          >
            <Dna className="h-4 w-4" />
            <span>scRNA-seq</span>
            <span className={cn(
              "ml-1 px-2 py-0.5 rounded-full text-xs",
              dataType === 'RNA' ? 'bg-white/20' : 'bg-[rgb(var(--muted))]'
            )}>
              {statsRNA.totalDatasets}
            </span>
          </button>
        </div>

        {/* Comparison Toggle */}
        <button
          onClick={() => setShowComparison(!showComparison)}
          className="text-sm text-[rgb(var(--primary))] hover:text-[rgb(var(--primary-hover))] transition-colors font-medium inline-flex items-center gap-2"
        >
          {showComparison ? (
            <>
              <span>✕</span>
              <span>Hide Comparison</span>
            </>
          ) : (
            <>
              <span>⚖️</span>
              <span>Compare Data Types</span>
            </>
          )}
        </button>
      </div>

      {/* Comparison Card (Optional) */}
      {showComparison && (
        <ComparisonCard statsATAC={statsATAC} statsRNA={statsRNA} />
      )}

      {/* Overview Cards */}
      <StatisticsOverview stats={currentStats} dataType={dataType} />

      {/* Summary Text */}
      <div className="card p-4 bg-[rgb(var(--info-bg))] border-[rgb(var(--info-border))] transition-colors">
        <p className="text-sm text-[rgb(var(--info-text))] transition-colors">
          📊 Currently viewing <strong>{currentStats.totalGSE} studies</strong> with{' '}
          <strong>{currentStats.totalDatasets} datasets</strong>, containing{' '}
          <strong>{currentStats.totalCells.toLocaleString()} cells</strong> and{' '}
          <strong>{currentStats.totalFeatures.toLocaleString()} {featureLabel}</strong>.
        </p>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Category Distribution */}
        <CategoryDistribution stats={currentStats} />

        {/* Organism Distribution */}
        <OrganismDistribution stats={currentStats} />

        {/* Platform Distribution */}
        <PlatformDistribution stats={currentStats} />

        {/* Cell Count Distribution */}
        <CellsDistribution gseGroups={currentGSEGroups} />
      </div>

      {/* Full Width Charts */}
      <div className="space-y-6">
        {/* Feature Distribution */}
        <FeaturesDistribution gseGroups={currentGSEGroups} dataType={dataType} />
      </div>
    </div>
  );
}