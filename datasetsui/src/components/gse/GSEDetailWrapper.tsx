'use client';

import { useSearchParams } from 'next/navigation';
import { getGSEGroup } from '@/lib/dataLoader';
import GSEDetail from './GSEDetail';
import { Suspense } from 'react';

interface GSEDetailWrapperProps {
  gseAccession: string;
}

function GSEDetailContent({ gseAccession }: GSEDetailWrapperProps) {
  const searchParams = useSearchParams();
  const type = (searchParams.get('type') as 'ATAC' | 'RNA') || 'ATAC';
  
  // Try to get the GSE group for the specified type
  let gseGroup = getGSEGroup(gseAccession, type);
  
  // If not found, try the other type
  if (!gseGroup) {
    const otherType = type === 'ATAC' ? 'RNA' : 'ATAC';
    gseGroup = getGSEGroup(gseAccession, otherType);
  }

  if (!gseGroup) {
    return (
      <div className="p-8 text-center">
        <h1 className="text-2xl font-bold text-[rgb(var(--foreground))] mb-4">Dataset Not Found</h1>
        <p className="text-[rgb(var(--muted-foreground))]">
          GSE accession {gseAccession} not found in our database.
        </p>
      </div>
    );
  }

  const actualType = gseGroup.datasets[0]?.dataFileName.includes('peak') ? 'ATAC' : 'RNA';

  return <GSEDetail gseGroup={gseGroup} dataType={actualType} />;
}

export default function GSEDetailWrapper({ gseAccession }: GSEDetailWrapperProps) {
  return (
    <Suspense fallback={<div className="p-8 text-center">Loading...</div>}>
      <GSEDetailContent gseAccession={gseAccession} />
    </Suspense>
  );
}
