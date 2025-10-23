'use client';

import { useEffect, useState } from 'react';
import { dataLoader } from '@/lib/dataLoader';
import type { DataManifest, SimulationResult } from '@/types/simulation';

export default function TestDataPage() {
  const [manifest, setManifest] = useState<DataManifest | null>(null);
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    setLoading(true);
    setError(null);

    try {
      // Test 1: Load manifest
      console.log('Loading manifest...');
      const manifestData = await dataLoader.loadManifest();
      setManifest(manifestData);
      console.log('✅ Manifest loaded:', manifestData);

      // Test 2: Load a simulation
      console.log('Loading simulation...');
      const sim = await dataLoader.loadSimulation('linear', 0.95, 0);
      setSimulation(sim);
      console.log('✅ Simulation loaded:', sim);

      // Test 3: Cache stats
      const stats = dataLoader.getCacheStats();
      console.log('📊 Cache stats:', stats);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      console.error('❌ Error:', err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container mx-auto p-8">
      <h1 className="text-3xl font-bold mb-6">Data Layer Test</h1>

      {loading && (
        <div className="text-blue-600 mb-4">Loading data...</div>
      )}

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <strong>Error:</strong> {error}
        </div>
      )}

      {manifest && (
        <div className="mb-8">
          <h2 className="text-2xl font-semibold mb-4">✅ Manifest Loaded</h2>
          <div className="bg-gray-100 p-4 rounded">
            <pre className="text-sm overflow-auto">
              {JSON.stringify(manifest, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {simulation && (
        <div className="mb-8">
          <h2 className="text-2xl font-semibold mb-4">✅ Simulation Loaded</h2>
          <div className="bg-gray-100 p-4 rounded">
            <p><strong>ID:</strong> {simulation.id}</p>
            <p><strong>Trajectory:</strong> {simulation.parameters.trajectory_type}</p>
            <p><strong>Continuity:</strong> {simulation.parameters.continuity}</p>
            <p><strong>N Cells:</strong> {simulation.metadata.n_cells}</p>
            <p><strong>Embeddings:</strong> {Object.keys(simulation.embeddings).join(', ')}</p>
            <p className="mt-2"><strong>Metrics:</strong></p>
            <pre className="text-xs overflow-auto mt-1">
              {JSON.stringify(simulation.metrics, null, 2)}
            </pre>
          </div>
        </div>
      )}

      <button
        onClick={loadData}
        disabled={loading}
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
      >
        Reload Data
      </button>
    </div>
  );
}