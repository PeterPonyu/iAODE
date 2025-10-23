import Link from 'next/link';
import { AlertCircle, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <AlertCircle className="h-16 w-16 text-gray-400 mb-4" />
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
        Dataset Not Found
      </h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6 text-center max-w-md">
        The GSE accession you're looking for doesn't exist in our database.
      </p>
      <Link href="/datasets" className="btn-primary inline-flex items-center gap-2">
        <ArrowLeft className="h-4 w-4" />
        Back to Browse
      </Link>
    </div>
  );
}