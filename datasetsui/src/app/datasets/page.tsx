import DatasetBrowserWrapper from '@/components/datasets/DatasetBrowserWrapper';

export const metadata = {
  title: 'Browse Datasets | iAODE-VAE Benchmark',
  description: 'Browse single-cell ATAC-seq and RNA-seq datasets from NCBI GEO',
};

export default function DatasetsPage() {
  return <DatasetBrowserWrapper />;
}