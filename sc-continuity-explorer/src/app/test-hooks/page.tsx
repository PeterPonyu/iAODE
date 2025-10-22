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
  const { theme, effectiveTheme, setTheme, toggleTheme } = useTheme();

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

      {/* Theme Test */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">🎨 useTheme</h2>
        <div className="space-y-2">
          <p>Current theme: <strong>{theme}</strong></p>
          <p>Effective theme: <strong>{effectiveTheme}</strong></p>
          <div className="flex gap-2">
            <button onClick={toggleTheme} className="btn btn-primary">
              Toggle Theme
            </button>
            <button onClick={() => setTheme('light')} className="btn btn-secondary">
              Light
            </button>
            <button onClick={() => setTheme('dark')} className="btn btn-secondary">
              Dark
            </button>
            <button onClick={() => setTheme('system')} className="btn btn-secondary">
              System
            </button>
          </div>
        </div>
      </div>

      {/* Debounce Test */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">⏱️ useDebounce</h2>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type something..."
          className="input mb-2"
        />
        <p>Immediate value: <strong>{inputValue}</strong></p>
        <p>Debounced value (500ms): <strong>{debouncedValue}</strong></p>
      </div>

      {/* Debounced Callback Test */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">📞 useDebouncedCallback</h2>
        <button
          onClick={() => debouncedLog(Math.random().toString())}
          className="btn btn-primary mb-2"
        >
          Click rapidly (debounced 500ms)
        </button>
        <div className="bg-gray-100 p-2 rounded text-sm space-y-1">
          {callbackLog.map((log, i) => (
            <div key={i}>{log}</div>
          ))}
        </div>
      </div>

      {/* LocalStorage Test */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">💾 useLocalStorage</h2>
        <p className="mb-2">Saved trajectory: <strong>{savedTrajectory}</strong></p>
        <div className="flex gap-2">
          <button onClick={() => setSavedTrajectory('linear')} className="btn btn-secondary">
            Linear
          </button>
          <button onClick={() => setSavedTrajectory('branching')} className="btn btn-secondary">
            Branching
          </button>
          <button onClick={() => setSavedTrajectory('cyclic')} className="btn btn-secondary">
            Cyclic
          </button>
        </div>
      </div>

      {/* Simulation Data Test */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">📊 useSimulationData</h2>
        {isLoading && <p>Loading simulation...</p>}
        {error && <p className="text-red-600">Error: {error}</p>}
        {simulation && (
          <div className="space-y-2">
            <p>✅ Simulation loaded!</p>
            <p>ID: <strong>{simulation.id}</strong></p>
            <p>Trajectory: <strong>{simulation.parameters.trajectory_type}</strong></p>
            <p>Continuity: <strong>{simulation.parameters.continuity}</strong></p>
            <p>Cells: <strong>{simulation.metadata.n_cells}</strong></p>
          </div>
        )}
      </div>
    </div>
  );
}