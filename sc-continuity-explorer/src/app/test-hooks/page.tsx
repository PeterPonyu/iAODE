'use client';

import { useState } from 'react';
import { useDebounce, useDebouncedCallback } from '@/hooks/useDebounce';
import { useTheme } from '@/hooks/useTheme';
import { useSimulationData } from '@/hooks/useSimulationData';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import type { TrajectoryType } from '@/types/simulation';

export default function TestHooksPage() {
  // Test useDebounce
  const [inputValue, setInputValue] = useState('');
  const debouncedValue = useDebounce(inputValue, 500);

  // Test useDebouncedCallback
  const [callbackLog, setCallbackLog] = useState<string[]>([]);
  const debouncedLog = useDebouncedCallback((value: string) => {
    setCallbackLog((prev) => [...prev, `Logged: ${value} at ${new Date().toLocaleTimeString()}`]);
  }, 500);

  // Test useTheme
  const { theme, effectiveTheme, setTheme, toggleTheme, mounted } = useTheme();

  // Test useLocalStorage
  const [savedTrajectory, setSavedTrajectory] = useLocalStorage<TrajectoryType>(
    'test-trajectory',
    'linear'
  );

  // Test useSimulationData
  const { simulation, isLoading, error } = useSimulationData({
    trajectoryType: 'linear',
    continuity: 0.95,
    autoLoad: true,
  });

  return (
    <div className="container mx-auto p-8 space-y-8">
      <h1 className="text-3xl font-bold">Hooks Test Page</h1>

      {/* Theme Test Card */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">🎨 useTheme</h2>
        <div className="space-y-3">
          {mounted ? (
            <div className="space-y-2">
              <p>Current theme: <strong>{theme}</strong></p>
              <p>Effective theme: <strong>{effectiveTheme}</strong></p>
              <div className="p-3 rounded border bg-muted">
                <p className="text-sm text-muted-foreground">
                  This box uses CSS variables that automatically switch themes!
                </p>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground">Loading theme...</p>
          )}
          
          <div className="flex gap-2 flex-wrap">
            <button 
              onClick={toggleTheme} 
              className="btn btn-primary"
              disabled={!mounted}
            >
              {mounted && (effectiveTheme === 'dark' ? '☀️' : '🌙')} Toggle Theme
            </button>
            <button 
              onClick={() => setTheme('light')} 
              className="btn btn-secondary"
              disabled={!mounted}
            >
              ☀️ Light
            </button>
            <button 
              onClick={() => setTheme('dark')} 
              className="btn btn-secondary"
              disabled={!mounted}
            >
              🌙 Dark
            </button>
            <button 
              onClick={() => setTheme('system')} 
              className="btn btn-secondary"
              disabled={!mounted}
            >
              💻 System
            </button>
          </div>
        </div>
      </div>

      {/* Debounce Test Card */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">⏱️ useDebounce</h2>
        <div className="space-y-3">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type something..."
            className="input w-full"
          />
          <div className="p-3 rounded bg-muted">
            <p className="text-sm">
              <span className="text-muted-foreground">Immediate value:</span>{' '}
              <strong>{inputValue || '(empty)'}</strong>
            </p>
            <p className="text-sm">
              <span className="text-muted-foreground">Debounced value (500ms):</span>{' '}
              <strong>{debouncedValue || '(empty)'}</strong>
            </p>
          </div>
        </div>
      </div>

      {/* Debounced Callback Test Card */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">📞 useDebouncedCallback</h2>
        <div className="space-y-3">
          <button
            onClick={() => debouncedLog(Math.random().toString())}
            className="btn btn-primary"
          >
            Click rapidly (debounced 500ms)
          </button>
          <div className="p-3 rounded border bg-muted max-h-40 overflow-auto">
            {callbackLog.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No logs yet. Click the button above!
              </p>
            ) : (
              <div className="space-y-1">
                {callbackLog.map((log, i) => (
                  <div key={i} className="text-xs font-mono p-1 rounded bg-background">
                    {log}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* LocalStorage Test Card */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">💾 useLocalStorage</h2>
        <div className="space-y-3">
          <div className="p-3 rounded bg-accent border">
            <p className="text-sm">
              Saved trajectory: <strong className="text-lg">{savedTrajectory}</strong>
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              This value persists across page refreshes
            </p>
          </div>
          <div className="flex gap-2 flex-wrap">
            <button 
              onClick={() => setSavedTrajectory('linear')} 
              className={savedTrajectory === 'linear' ? 'btn btn-primary' : 'btn btn-secondary'}
            >
              Linear
            </button>
            <button 
              onClick={() => setSavedTrajectory('branching')} 
              className={savedTrajectory === 'branching' ? 'btn btn-primary' : 'btn btn-secondary'}
            >
              Branching
            </button>
            <button 
              onClick={() => setSavedTrajectory('cyclic')} 
              className={savedTrajectory === 'cyclic' ? 'btn btn-primary' : 'btn btn-secondary'}
            >
              Cyclic
            </button>
            <button 
              onClick={() => setSavedTrajectory('discrete')} 
              className={savedTrajectory === 'discrete' ? 'btn btn-primary' : 'btn btn-secondary'}
            >
              Discrete
            </button>
          </div>
        </div>
      </div>

      {/* Simulation Data Test Card */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">📊 useSimulationData</h2>
        <div className="space-y-3">
          {isLoading && (
            <div className="flex items-center gap-3 p-4 rounded bg-muted">
              <div className="spinner" />
              <p>Loading simulation...</p>
            </div>
          )}
          
          {error && (
            <div className="p-4 rounded border-2" style={{ 
              backgroundColor: 'var(--color-primary-50)', 
              borderColor: 'var(--color-primary-500)',
              color: 'var(--color-primary-900)' 
            }}>
              <p className="font-semibold">❌ Error</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          )}
          
          {simulation && !isLoading && (
            <div className="space-y-3">
              <div className="p-3 rounded border-2" style={{ 
                backgroundColor: 'var(--color-primary-50)', 
                borderColor: 'var(--color-primary-500)',
                color: 'var(--color-primary-900)' 
              }}>
                <p className="font-semibold">✅ Simulation loaded successfully!</p>
              </div>
              
              <div className="p-4 rounded border bg-muted">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-muted-foreground">ID</p>
                    <p className="font-mono text-xs mt-1">{simulation.id}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Trajectory</p>
                    <p className="font-semibold mt-1">{simulation.parameters.trajectory_type}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Continuity</p>
                    <p className="font-semibold mt-1">{simulation.parameters.continuity.toFixed(3)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Cells</p>
                    <p className="font-semibold mt-1">{simulation.metadata.n_cells}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Dimensions</p>
                    <p className="font-semibold mt-1">{simulation.metadata.n_dims}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Embeddings</p>
                    <p className="font-semibold mt-1">{Object.keys(simulation.embeddings).join(', ')}</p>
                  </div>
                </div>
              </div>

              {/* Metrics Preview */}
              <div className="p-4 rounded border">
                <p className="text-sm font-semibold mb-2">Core Metrics</p>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  {Object.entries(simulation.metrics)
                    .slice(0, 6)
                    .map(([key, value]) => (
                      <div key={key} className="p-2 rounded bg-muted">
                        <p className="text-muted-foreground truncate">
                          {key.replace(/_/g, ' ')}
                        </p>
                        <p className="font-mono mt-1">
                          {typeof value === 'number' ? value.toFixed(3) : String(value)}
                        </p>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Color Palette Demo */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">🎨 CSS Variable Color Palette</h2>
        <p className="text-sm text-muted-foreground mb-4">
          All colors automatically switch with theme changes
        </p>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div className="p-3 rounded border bg-background">
            <p className="text-xs font-mono">Background</p>
            <p className="text-xs text-muted-foreground mt-1">bg-background</p>
          </div>
          <div className="p-3 rounded border bg-foreground text-background">
            <p className="text-xs font-mono">Foreground</p>
            <p className="text-xs opacity-70 mt-1">bg-foreground</p>
          </div>
          <div className="p-3 rounded border bg-card">
            <p className="text-xs font-mono">Card</p>
            <p className="text-xs text-muted-foreground mt-1">bg-card</p>
          </div>
          <div className="p-3 rounded border bg-muted">
            <p className="text-xs font-mono">Muted</p>
            <p className="text-xs text-muted-foreground mt-1">bg-muted</p>
          </div>
          <div className="p-3 rounded border bg-accent">
            <p className="text-xs font-mono">Accent</p>
            <p className="text-xs text-muted-foreground mt-1">bg-accent</p>
          </div>
          <div className="p-3 rounded border-2">
            <p className="text-xs font-mono">Border</p>
            <p className="text-xs text-muted-foreground mt-1">border</p>
          </div>
        </div>
      </div>
    </div>
  );
}