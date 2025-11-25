import { getUniqueGSEAccessions } from '@/lib/dataLoader';
import GSEDetailWrapper from '@/components/gse/GSEDetailWrapper';

interface PageProps {
  params: Promise<{
    gseAccession: string;
  }>;
}

// Generate static paths for all GSE accessions
export async function generateStaticParams() {
  const gseAccessionsATAC = getUniqueGSEAccessions('ATAC');
  const gseAccessionsRNA = getUniqueGSEAccessions('RNA');
  
  // Get unique accessions from both types
  const allAccessions = Array.from(new Set([
    ...gseAccessionsATAC,
    ...gseAccessionsRNA
  ]));
  
  return allAccessions.map(acc => ({ gseAccession: acc }));
}

export default async function GSEDetailPage({ params }: PageProps) {
  const { gseAccession } = await params;
  
  return <GSEDetailWrapper gseAccession={gseAccession} />;
}