'use client';

import { useState } from 'react';
import { loadSimulation, loadManifest, clearCache } from '@/lib/dataLoader';
import { TrajectoryType } from '@/types/simulation';
import { Button, Card, CardHeader, CardTitle, CardContent, Alert } from '@/components/ui';

export default function TestDataPage() {
  const [result, setResult] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const testLoadManifest = async () => {
    setIsLoading(true);
    try {
      const manifest = await loadManifest();
      setResult(JSON.stringify(manifest, null, 2));
    } catch (error) {
      setResult(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const testLoadSimulation = async (trajectory: TrajectoryType, continuity: number) => {
    setIsLoading(true);
    try {
      const simulation = await loadSimulation(trajectory, continuity, 0);
      setResult(JSON.stringify({
        id: simulation.id,
        parameters: simulation.parameters,
        embeddings: Object.keys(simulation.embeddings),
        metadata: {
          n_cells: simulation.metadata.n_cells,
          n_dims: simulation.metadata.n_dims,
        },
        metricsKeys: Object.keys(simulation.metrics),
      }, null, 2));
    } catch (error) {
      setResult(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-8 space-y-6">
      <h1 className="text-3xl font-bold">Data Loading Test</h1>

      <Card>
        <CardHeader>
          <CardTitle>Test Functions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button onClick={testLoadManifest} disabled={isLoading}>
              Load Manifest
            </Button>
            <Button onClick={() => testLoadSimulation('linear', 0.95)} disabled={isLoading}>
              Load Linear 95%
            </Button>
            <Button onClick={() => testLoadSimulation('branching', 0.90)} disabled={isLoading}>
              Load Branching 90%
            </Button>
            <Button onClick={() => { clearCache(); setResult('Cache cleared'); }}>
              Clear Cache
            </Button>
          </div>

          {isLoading && <Alert variant="info">Loading...</Alert>}

          {result && (
            <div className="mt-4">
              <p className="text-sm font-semibold mb-2">Result:</p>
              <pre className="bg-muted p-4 rounded text-xs overflow-auto max-h-96">
                {result}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>

      <Alert variant="info" title="Expected Files">
        <ul className="text-sm space-y-1">
          <li>✓ /public/data/manifest.json</li>
          <li>✓ /public/data/metadata/parameter_lookup.json</li>
          <li>✓ /public/data/chunks/chunk_0.json</li>
          <li>✓ /public/data/chunks/chunk_1.json</li>
        </ul>
      </Alert>
    </div>
  );
}