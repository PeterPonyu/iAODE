import { GSEGroup } from '@/types/datasets';
import GSECard from './GSECard';

interface DatasetGridProps {
  data: GSEGroup[];
  dataType: 'ATAC' | 'RNA';
}

export default function DatasetGrid({ data, dataType }: DatasetGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
      {data.map((gseGroup) => (
        <GSECard 
          key={gseGroup.gseAccession} 
          gseGroup={gseGroup}
          dataType={dataType}
        />
      ))}
    </div>
  );
}