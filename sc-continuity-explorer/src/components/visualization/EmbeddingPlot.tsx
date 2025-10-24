'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { SimulationData } from '@/types/simulation';
import { getEmbedding } from '@/lib/dataUtils';
import { Card } from '@/components/ui';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export interface EmbeddingPlotProps {
  simulation: SimulationData;
  embeddingMethod: string;
  colorBy: 'pseudotime' | 'cell_types';
  width?: number;
  height?: number;
  className?: string;
}

export function EmbeddingPlot({
  simulation,
  embeddingMethod,
  colorBy = 'pseudotime',
  width = 600,
  height = 600,
  className = '',
}: EmbeddingPlotProps) {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const embedding = getEmbedding(simulation, embeddingMethod);

  if (!embedding || embedding.length === 0) {
    return (
      <Card className={className}>
        <div className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">No embedding data available</p>
        </div>
      </Card>
    );
  }

  if (!isClient) {
    return (
      <Card className={className}>
        <div className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Loading visualization...</p>
        </div>
      </Card>
    );
  }

  // Extract x, y coordinates
  const x = embedding.map((point) => point[0]);
  const y = embedding.map((point) => point[1]);

  // Get color values
  const colorValues = colorBy === 'pseudotime'
    ? simulation.metadata.pseudotime
    : simulation.metadata.cell_types;

  const trace: any = {
    x,
    y,
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: 4,
      color: colorValues,
      colorscale: colorBy === 'pseudotime' ? 'Viridis' : undefined,
      showscale: colorBy === 'pseudotime',
      colorbar: colorBy === 'pseudotime' ? {
        title: { text: 'Pseudotime' },
        thickness: 15,
      } : undefined,
    },
    text: colorValues.map((v, i) => `Cell ${i}: ${v}`),
    hovertemplate: '<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
  };

  const layout: any = {
    title: {
      text: `${embeddingMethod.toUpperCase()} Embedding`,
      font: { size: 16 },
    },
    xaxis: { 
      title: { text: 'Component 1' },
      showgrid: true, 
      zeroline: false 
    },
    yaxis: { 
      title: { text: 'Component 2' },
      showgrid: true, 
      zeroline: false 
    },
    width,
    height,
    hovermode: 'closest',
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    margin: { l: 60, r: 40, t: 60, b: 60 },
  };

  const config: any = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'] as any,
  };

  return (
    <Card className={className}>
      <Plot data={[trace]} layout={layout} config={config} />
    </Card>
  );
}