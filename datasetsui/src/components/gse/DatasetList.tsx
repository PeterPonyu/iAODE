import { MergedDataset } from '@/types/datasets';
import DatasetCard from './DatasetCard';

interface DatasetListProps {
  datasets: MergedDataset[];
}

export default function DatasetList({ datasets }: DatasetListProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {datasets.map((dataset) => (
        <DatasetCard key={dataset.id} dataset={dataset} />
      ))}
    </div>
  );
}