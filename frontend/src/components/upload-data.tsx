
'use client';

import { useState } from 'react';
import { uploadData } from '@/lib/api';
import { DataInfo, DataType } from '@/lib/types';

type UploadDataProps = {
  onUploadSuccess: (info: DataInfo) => void;
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
      onUploadSuccess(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-4 p-6 rounded-lg card">
      <h2 className="text-xl font-semibold">Upload Data</h2>
      
      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium mb-2">
            Data Type
          </label>
          <select
            value={dataType}
            onChange={(e) => setDataType(e.target.value as DataType)}
            className="w-full px-3 py-2 rounded-lg"
          >
            <option value="scrna">scRNA-seq</option>
            <option value="scatac">scATAC-seq</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Select File (.h5ad or .h5)
          </label>
          <input
            type="file"
            accept=".h5ad,.h5"
            onChange={handleFileChange}
            className="w-full px-3 py-2 rounded-lg"
          />
          {file && (
            <p className="mt-2 text-sm text-muted">
              Selected: {file.name}
            </p>
          )}
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="w-full px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 btn-primary"
        >
          {uploading ? 'Uploading...' : 'Upload'}
        </button>

        {error && (
          <p className="text-sm" style={{ color: 'var(--color-error-text)' }}>{error}</p>
        )}
      </div>
    </div>
  );
}
