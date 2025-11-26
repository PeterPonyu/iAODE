// ============================================================================
// components/upload-data.tsx - REFINED
// ============================================================================

'use client';

import { useState } from 'react';
import { uploadData } from '@/lib/api';
import { DataInfo, DataType } from '@/lib/types';

type UploadDataProps = {
  onUploadSuccess: (info: DataInfo, dataType: DataType) => void;
};

export function UploadData({ onUploadSuccess }: UploadDataProps) {
  const [file, setFile] = useState<File | null>(null);
  const [dataType, setDataType] = useState<DataType>('scrna');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const info = await uploadData(file, dataType);
      onUploadSuccess(info, dataType);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="card rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Upload Data</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2 text-muted">
            Data Type
          </label>
          <select
            value={dataType}
            onChange={(e) => setDataType(e.target.value as DataType)}
            className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--background))] text-[rgb(var(--foreground))] text-sm focus:outline-none focus:ring-2 focus:ring-[rgb(var(--primary))] disabled:opacity-50"
            disabled={uploading}
          >
            <option value="scrna" className="bg-[rgb(var(--background))] text-[rgb(var(--foreground))]">scRNA-seq</option>
            <option value="scatac" className="bg-[rgb(var(--background))] text-[rgb(var(--foreground))]">scATAC-seq</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2 text-muted">
            Select File
          </label>
          <input
            type="file"
            accept=".h5ad,.h5"
            onChange={handleFileChange}
            disabled={uploading}
            className="w-full px-3 py-2 rounded-lg border text-sm file:mr-4 file:py-1 file:px-3 file:rounded file:border-0 file:text-sm file:font-medium file:btn-secondary"
          />
          {file && (
            <p className="mt-2 text-sm text-muted">
              Selected: <span className="font-medium">{file.name}</span>
            </p>
          )}
        </div>

        {error && (
          <div className="p-3 rounded-lg badge-error">
            <p className="text-sm font-medium">{error}</p>
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="w-full px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed btn-primary"
        >
          {uploading ? 'Uploading...' : 'Upload Data'}
        </button>
      </div>
    </div>
  );
}
