// ============================================================================
// app/train/page.tsx - REFINED
// ============================================================================

'use client';

import { useState } from 'react';
import { Header } from '@/components/header';
import { UploadData } from '@/components/upload-data';
import { PreprocessingPanel } from '@/components/preprocessing-panel';
import { TrainingParams } from '@/components/training-params';
import { TrainingMonitor } from '@/components/training-monitor';
import { startTraining } from '@/lib/api';
import { DataInfo, AgentParams, TrainParams, DataType, PreprocessInfo } from '@/lib/types';

export default function TrainPage() {
  const [dataInfo, setDataInfo] = useState<DataInfo | null>(null);
  const [dataType, setDataType] = useState<DataType>('scrna');
  const [preprocessingComplete, setPreprocessingComplete] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  const handleUploadSuccess = (info: DataInfo, type: DataType) => {
    setDataInfo(info);
    setDataType(type);
    setPreprocessingComplete(type === 'scrna'); // scRNA doesn't need preprocessing
  };

  const handlePreprocessingComplete = (info: PreprocessInfo) => {
    setDataInfo({ n_cells: info.n_cells, n_genes: info.n_peaks });
    setPreprocessingComplete(true);
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
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-3xl font-bold mb-2 text-[rgb(var(--text-primary))]">Model Training</h1>
        <p className="text-[rgb(var(--muted-foreground))] mb-8">
          Upload your data, configure preprocessing (if needed), and train the model
        </p>

        {/* Data Status Banner */}
        {dataInfo && (
          <div className="mb-6 p-4 rounded-lg badge-info">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium mb-1">
                  ✓ {dataType === 'scatac' ? 'scATAC-seq' : 'scRNA-seq'} Data Loaded
                </p>
                <p className="text-sm">
                  {dataInfo.n_cells.toLocaleString()} cells × {dataInfo.n_genes.toLocaleString()} {dataType === 'scatac' ? 'peaks' : 'genes'}
                </p>
              </div>
              {dataType === 'scatac' && !preprocessingComplete && (
                <span className="text-xs badge-purple px-3 py-1.5 rounded-full">
                  Preprocessing Required
                </span>
              )}
              {preprocessingComplete && (
                <span className="text-xs badge-success px-3 py-1.5 rounded-full">
                  Ready to Train
                </span>
              )}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Upload & Config */}
          <div className="lg:col-span-2 space-y-6">
            <UploadData onUploadSuccess={handleUploadSuccess} />
            
            {dataInfo && dataType === 'scatac' && !preprocessingComplete && (
              <PreprocessingPanel 
                dataType={dataType}
                onComplete={handlePreprocessingComplete}
              />
            )}

            {dataInfo && preprocessingComplete && (
              <TrainingParams
                onSubmit={handleStartTraining}
                disabled={isTraining}
              />
            )}
          </div>

          {/* Right Column: Monitor */}
          <div>
            <TrainingMonitor />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[rgb(var(--border))] mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-sm text-[rgb(var(--muted-foreground))]">
          iAODE Training Interface © {new Date().getFullYear()}
        </div>
      </footer>
    </div>
  );
}
