
'use client';

import { useEffect, useState } from 'react';
import { getTrainingState, downloadEmbedding } from '@/lib/api';
import { TrainingState } from '@/lib/types';

export function TrainingMonitor() {
  const [state, setState] = useState<TrainingState | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  useEffect(() => {
    const fetchState = async () => {
      try {
        const data = await getTrainingState();
        setState(data);
      } catch {
        // Backend may not be ready yet, set a waiting state
        // This is expected during initial load before the backend is running
        if (!state) {
          setState({
            status: 'idle',
            current_epoch: 0,
            message: 'Waiting for backend connection...'
          });
        }
      }
    };

    // Initial fetch
    fetchState();

    // Dynamic polling interval based on training status
    const getPollingInterval = () => {
      if (!state) return 5000; // 5s when initializing
      if (state.status === 'training' || state.status === 'initializing') return 3000; // 3s during training
      if (state.status === 'completed' || state.status === 'error') return 10000; // 10s when done
      return 5000; // 5s for idle
    };

    const interval = setInterval(fetchState, getPollingInterval());

    return () => clearInterval(interval);
  }, [state?.status]); // Re-create interval when status changes

  const handleDownload = async (type: 'latent' | 'interpretable') => {
    setDownloading(type);
    try {
      await downloadEmbedding(type);
    } catch (err) {
      alert(`Download failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setDownloading(null);
    }
  };

  if (!state) {
    return (
      <div className="card">
        <p className="text-[rgb(var(--muted-foreground))]">Loading state...</p>
      </div>
    );
  }

  const isTraining = state.status === 'training' || state.status === 'initializing';
  const isCompleted = state.status === 'completed';

  return (
    <div className="card space-y-4">
      <h2 className="text-xl font-semibold text-[rgb(var(--text-primary))]">Training Status</h2>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-[rgb(var(--text-secondary))]">Status:</span>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            isCompleted ? 'badge-success' :
            isTraining ? 'badge-info' :
            state.status === 'error' ? 'badge-error' :
            'badge-idle'
          }`}>
            {state.status}
          </span>
        </div>

        {state.current_epoch > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-[rgb(var(--text-secondary))]">Current Epoch:</span>
            <span className="text-sm text-[rgb(var(--foreground))]">{state.current_epoch}</span>
          </div>
        )}

        {state.message && (
          <div className="p-3 rounded-lg card-inner">
            <p className="text-sm text-[rgb(var(--foreground))]">{state.message}</p>
          </div>
        )}

        {isTraining && (
          <div className="w-full rounded-full h-2 overflow-hidden progress-bar">
            <div className="h-full animate-pulse progress-fill" />
          </div>
        )}

        {isCompleted && (
          <div className="space-y-2 pt-4 border-t-divider">
            <h3 className="text-sm font-medium mb-2 text-[rgb(var(--text-secondary))]">Download Embeddings</h3>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => handleDownload('latent')}
                disabled={downloading !== null}
                className="btn-primary px-4 py-2 rounded-lg text-sm font-medium"
              >
                {downloading === 'latent' ? 'Downloading...' : 'Latent'}
              </button>
              <button
                onClick={() => handleDownload('interpretable')}
                disabled={downloading !== null}
                className="btn-purple px-4 py-2 rounded-lg text-sm font-medium"
              >
                {downloading === 'interpretable' ? 'Downloading...' : 'Interpretable'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
