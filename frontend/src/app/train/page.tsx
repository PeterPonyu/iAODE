
'use client';

import { useState } from 'react';
import { Header } from '@/components/header';
import { UploadData } from '@/components/upload-data';
import { TrainingParams } from '@/components/training-params';
import { TrainingMonitor } from '@/components/training-monitor';
import { startTraining } from '@/lib/api';
import { DataInfo, AgentParams, TrainParams } from '@/lib/types';

export default function TrainPage() {
  const [dataInfo, setDataInfo] = useState<DataInfo | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  const handleUploadSuccess = (info: DataInfo) => {
    setDataInfo(info);
  };

  const handleStartTraining = async (agentParams: AgentParams, trainParams: TrainParams) => {
    setIsTraining(true);
    try {
      await startTraining(agentParams, trainParams);
    } catch (err) {
      alert(`Training failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setIsTraining(false);
    }
  };

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--color-surface-secondary)' }}>
      <Header />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-3xl font-bold mb-8">Model Training</h1>

        {dataInfo && (
          <div className="mb-6 p-4 rounded-lg" style={{
            backgroundColor: 'var(--color-info)',
            borderWidth: '1px',
            borderColor: 'var(--color-info-border)'
          }}>
            <p className="text-sm" style={{ color: 'var(--color-info-text)' }}>
              ✓ Data loaded: {dataInfo.n_cells.toLocaleString()} cells × {dataInfo.n_genes.toLocaleString()} genes
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <UploadData onUploadSuccess={handleUploadSuccess} />
            
            {dataInfo && (
              <TrainingParams
                onSubmit={handleStartTraining}
                disabled={isTraining}
              />
            )}
          </div>

          <div>
            <TrainingMonitor />
          </div>
        </div>
      </main>
    </div>
  );
}
