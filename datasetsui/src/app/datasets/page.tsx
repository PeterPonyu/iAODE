import { getAllGSEGroups } from '@/lib/dataLoader';
import DatasetBrowser from '@/components/datasets/DatasetBrowser';

export const metadata = {
  title: 'Browse Datasets | scATAC-seq Browser',
  description: 'Browse and search single-cell ATAC-seq datasets from NCBI GEO',
};

export default function DatasetsPage() {
  const gseGroups = getAllGSEGroups();
  const totalDatasets = gseGroups.reduce((sum, g) => sum + g.datasets.length, 0);
  
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">
          Browse Datasets
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Explore {gseGroups.length} studies with {totalDatasets} datasets from NCBI GEO
        </p>
      </div>

      {/* Main Browser Component */}
      <DatasetBrowser initialData={gseGroups} />
    </div>
  );
}