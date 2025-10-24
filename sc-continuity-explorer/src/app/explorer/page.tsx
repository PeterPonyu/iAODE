'use client';

import { useState } from 'react';
import { useSimulationData } from '@/hooks/useSimulationData';
import { TrajectoryType } from '@/types/simulation';
import {
  ContinuitySlider,
  TrajectorySelector,
  EmbeddingSelector,
  ColorBySelector,
} from '@/components/controls';
import { EmbeddingPlot } from '@/components/visualization/EmbeddingPlot';
import { Alert } from '@/components/ui';

export default function ExplorerPage() {
  const [trajectoryType, setTrajectoryType] = useState<TrajectoryType>('linear');
  const [continuity, setContinuity] = useState(0.95);
  const [embeddingMethod, setEmbeddingMethod] = useState('umap');
  const [colorBy, setColorBy] = useState<'pseudotime' | 'cell_types'>('pseudotime');

  const { simulation, isLoading, error } = useSimulationData({
    trajectoryType,
    continuity,
    replicate: 1,
    autoLoad: true,
  });

  return (
    <div className="container mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold">Single-Cell Continuity Explorer</h1>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <TrajectorySelector value={trajectoryType} onChange={setTrajectoryType} />
        <ContinuitySlider value={continuity} onChange={setContinuity} />
        <EmbeddingSelector
          simulation={simulation}
          value={embeddingMethod}
          onChange={setEmbeddingMethod}
        />
        <ColorBySelector value={colorBy} onChange={setColorBy} />
      </div>

      {/* Loading/Error States */}
      {isLoading && <Alert variant="info">Loading simulation...</Alert>}
      {error && <Alert variant="error">{error}</Alert>}

      {/* Visualization */}
      {simulation && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <EmbeddingPlot
              simulation={simulation}
              embeddingMethod={embeddingMethod}
              colorBy={colorBy}
            />
          </div>
        </div>
      )}
    </div>
  );
}