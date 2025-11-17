import { MergedDataset } from '@/types/datasets';
import DatasetCard from './DatasetCard';

interface DatasetListProps {
  datasets: MergedDataset[];
  dataType: 'ATAC' | 'RNA';
}

export default function DatasetList({ datasets, dataType }: DatasetListProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {datasets.map((dataset) => (
        <DatasetCard key={dataset.id} dataset={dataset} dataType={dataType} />
      ))}
    </div>
  );
}