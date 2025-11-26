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
    <div className="card">
      <h2 className="text-xl font-semibold mb-4 text-[rgb(var(--text-primary))]">Upload Data</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">
            Data Type
          </label>
          <select
            value={dataType}
            onChange={(e) => setDataType(e.target.value as DataType)}
            className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
            disabled={uploading}
          >
            <option value="scrna">scRNA-seq</option>
            <option value="scatac">scATAC-seq</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">
            Select File
          </label>
          <input
            type="file"
            accept=".h5ad,.h5"
            onChange={handleFileChange}
            disabled={uploading}
            className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))] file:mr-4 file:py-1 file:px-3 file:rounded file:border-0 file:text-sm file:font-medium file:bg-[rgb(var(--secondary))] file:text-[rgb(var(--secondary-foreground))]"
          />
          {file && (
            <p className="mt-2 text-sm text-[rgb(var(--muted-foreground))]">
              Selected: <span className="font-medium text-[rgb(var(--foreground))]">{file.name}</span>
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
          className="btn-primary w-full"
        >
          {uploading ? 'Uploading...' : 'Upload Data'}
        </button>
      </div>
    </div>
  );
}
