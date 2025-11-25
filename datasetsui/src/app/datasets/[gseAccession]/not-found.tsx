import Link from 'next/link';
import { AlertCircle, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <AlertCircle className="h-16 w-16 text-[rgb(var(--text-muted))] mb-4 transition-colors" />
      <h1 className="text-2xl font-bold text-[rgb(var(--foreground))] mb-2 transition-colors">
        Dataset Not Found
      </h1>
      <p className="text-[rgb(var(--muted-foreground))] mb-6 text-center max-w-md transition-colors">
        The GSE accession you&apos;re looking for doesn&apos;t exist in our database.
      </p>
      <Link href="/datasets" className="btn-primary inline-flex items-center gap-2">
        <ArrowLeft className="h-4 w-4" />
        Back to Browse
      </Link>
    </div>
  );
}