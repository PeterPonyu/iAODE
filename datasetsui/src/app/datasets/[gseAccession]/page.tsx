import { notFound } from 'next/navigation';
import { getGSEGroup, getUniqueGSEAccessions } from '@/lib/dataLoader';
import GSEDetail from '@/components/gse/GSEDetail';
import { Metadata } from 'next';

interface PageProps {
  params: Promise<{  // ← Add Promise wrapper
    gseAccession: string;
  }>;
}

// Generate static paths for all GSE accessions
export async function generateStaticParams() {
  const gseAccessions = getUniqueGSEAccessions();
  return gseAccessions.map((gseAccession) => ({
    gseAccession,
  }));
}

// Generate metadata
export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { gseAccession } = await params; // ← Add await here
  const gseGroup = getGSEGroup(gseAccession);
  
  if (!gseGroup) {
    return {
      title: 'Dataset Not Found',
    };
  }

  return {
    title: `${gseGroup.gseAccession} - ${gseGroup.title}`,
    description: `${gseGroup.datasets.length} datasets from ${gseGroup.authors}. ${gseGroup.organism} scATAC-seq data.`,
  };
}

export default async function GSEDetailPage({ params }: PageProps) {
  const { gseAccession } = await params; // ← Add await here
  const gseGroup = getGSEGroup(gseAccession);

  if (!gseGroup) {
    notFound();
  }

  return <GSEDetail gseGroup={gseGroup} />;
}