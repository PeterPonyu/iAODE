
'use client';

import { useEffect, useState } from 'react';
import { getTrainingState, downloadEmbedding } from '@/lib/api';
import { TrainingState } from '@/lib/types';

export function TrainingMonitor() {
  const [state, setState] = useState<TrainingState | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchState = async () => {
      try {
        const data = await getTrainingState();
        setState(data);
        setError(null);
      } catch (err) {
        // Silently handle error if backend not ready
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

    // Poll every 2 seconds
    const interval = setInterval(fetchState, 2000);

    return () => clearInterval(interval);
  }, [state]);

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
      <div className="space-y-4 p-6 rounded-lg card">
        <p className="text-muted">Loading state...</p>
      </div>
    );
  }

  const isTraining = state.status === 'training' || state.status === 'initializing';
  const isCompleted = state.status === 'completed';
  const isIdle = state.status === 'idle';

  return (
    <div className="space-y-4 p-6 rounded-lg card">
      <h2 className="text-xl font-semibold">Training Status</h2>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Status:</span>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            isCompleted ? 'badge-status-success' :
            isTraining ? 'badge-status-info' :
            state.status === 'error' ? 'badge-status-error' :
            'badge-status-idle'
          }`}>
            {state.status}
          </span>
        </div>

        {state.current_epoch > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Current Epoch:</span>
            <span className="text-sm">{state.current_epoch}</span>
          </div>
        )}

        {state.message && (
          <div className="p-3 rounded-lg" style={{
            backgroundColor: 'var(--color-card)',
            border: '1px solid var(--color-border)'
          }}>
            <p className="text-sm">{state.message}</p>
          </div>
        )}

        {isTraining && (
          <div className="w-full rounded-full h-2 overflow-hidden" style={{ backgroundColor: 'var(--color-secondary)' }}>
            <div className="h-full animate-pulse" style={{ 
              backgroundColor: 'var(--color-primary)',
              width: '100%'
            }} />
          </div>
        )}

        {isCompleted && (
          <div className="space-y-2 pt-4" style={{ borderTop: '1px solid var(--color-border)' }}>
            <h3 className="text-sm font-medium mb-2">Download Embeddings</h3>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => handleDownload('latent')}
                disabled={downloading !== null}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 btn-primary"
              >
                {downloading === 'latent' ? 'Downloading...' : 'Latent'}
              </button>
              <button
                onClick={() => handleDownload('interpretable')}
                disabled={downloading !== null}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                style={{
                  backgroundColor: 'var(--color-purple-text)',
                  color: 'white'
                }}
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
