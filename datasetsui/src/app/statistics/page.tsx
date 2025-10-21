import { getAllGSEGroups } from '@/lib/dataLoader';
import { calculateStats } from '@/lib/statsCalculator';
import StatisticsOverview from '@/components/statistics/StatisticsOverview';
import CategoryDistribution from '@/components/statistics/CategoryDistribution';
import OrganismDistribution from '@/components/statistics/OrganismDistribution';
import PlatformDistribution from '@/components/statistics/PlatformDistribution';
import CellsDistribution from '@/components/statistics/CellsDistribution';
import PeaksDistribution from '@/components/statistics/PeaksDistribution';

export const metadata = {
  title: 'Statistics | scATAC-seq Browser',
  description: 'Statistical analysis and visualizations of scATAC-seq datasets',
};

export default function StatisticsPage() {
  const gseGroups = getAllGSEGroups();
  const stats = calculateStats(gseGroups);

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-[rgb(var(--foreground))] mb-2 transition-colors">
          Dataset Statistics
        </h1>
        <p className="text-[rgb(var(--muted-foreground))] transition-colors">
          Comprehensive analysis and visualizations of {stats.totalGSE} studies 
          with {stats.totalDatasets} datasets
        </p>
      </div>

      {/* Overview Cards */}
      <StatisticsOverview stats={stats} />

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Category Distribution */}
        <CategoryDistribution stats={stats} />

        {/* Organism Distribution */}
        <OrganismDistribution stats={stats} />

        {/* Platform Distribution */}
        <PlatformDistribution stats={stats} />

        {/* Cell Count Distribution */}
        <CellsDistribution gseGroups={gseGroups} />
      </div>

      {/* Full Width Charts */}
      <div className="space-y-6">
        {/* Peak Distribution */}
        <PeaksDistribution gseGroups={gseGroups} />
      </div>
    </div>
  );
}