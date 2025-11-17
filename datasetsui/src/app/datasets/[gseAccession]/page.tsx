import { notFound } from 'next/navigation';
import { getGSEGroup, getUniqueGSEAccessions } from '@/lib/dataLoader';
import GSEDetail from '@/components/gse/GSEDetail';
import { Metadata } from 'next';

interface PageProps {
  params: Promise<{
    gseAccession: string;
  }>;
  searchParams: Promise<{
    type?: 'ATAC' | 'RNA';
  }>;
}

// Generate static paths for all GSE accessions with both types
export async function generateStaticParams() {
  const gseAccessionsATAC = getUniqueGSEAccessions('ATAC');
  const gseAccessionsRNA = getUniqueGSEAccessions('RNA');
  
  return [
    ...gseAccessionsATAC.map(acc => ({ gseAccession: acc })),
    ...gseAccessionsRNA.map(acc => ({ gseAccession: acc })),
  ];
}

// Generate metadata
export async function generateMetadata({ params, searchParams }: PageProps): Promise<Metadata> {
  const { gseAccession } = await params;
  const { type = 'ATAC' } = await searchParams;
  
  const gseGroup = getGSEGroup(gseAccession, type);
  
  if (!gseGroup) {
    return {
      title: 'Dataset Not Found',
    };
  }

  const dataLabel = type === 'ATAC' ? 'scATAC-seq' : 'scRNA-seq';
  const featureLabel = type === 'ATAC' ? 'peaks' : 'genes';

  return {
    title: `${gseGroup.gseAccession} - ${gseGroup.title} | iAODE-VAE`,
    description: `${gseGroup.datasets.length} ${dataLabel} datasets from ${gseGroup.authors}. ${gseGroup.organism} data with ${gseGroup.totalCells.toLocaleString()} cells and ${gseGroup.totalFeatures.toLocaleString()} ${featureLabel} in standardized 10X h5 format.`,
  };
}

export default async function GSEDetailPage({ params, searchParams }: PageProps) {
  const { gseAccession } = await params;
  const { type = 'ATAC' } = await searchParams;
  
  const gseGroup = getGSEGroup(gseAccession, type);

  if (!gseGroup) {
    notFound();
  }

  return <GSEDetail gseGroup={gseGroup} dataType={type} />;
}