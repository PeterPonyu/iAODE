// ============================================================================
// FILE: components/EmbeddingPlot.tsx (FIXED - Cell Type Coloring + Dynamic Config)
// ============================================================================

'use client';

import { useMemo } from 'react';
import dynamic from 'next/dynamic';
import { SimulationResult, EmbeddingMethod, ColorByOption } from '@/types/simulation';
import { getEmbedding } from '@/lib/dataUtils';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

type EmbeddingPlotProps = {
  simulation: SimulationResult;
  embeddingMethod: EmbeddingMethod;
  colorBy: ColorByOption;
};

export function EmbeddingPlot({ simulation, embeddingMethod, colorBy }: EmbeddingPlotProps) {
  const embedding = getEmbedding(simulation, embeddingMethod);

  const { data, layout } = useMemo(() => {
    if (!embedding || embedding.length === 0) {
      return { data: [], layout: {} };
    }

    const x = embedding.map((point) => point[0]);
    const y = embedding.map((point) => point[1]);

    // ✅ FIX: Better color value handling with type checking
    const getColorValues = (): { values: number[] | string[]; isNumeric: boolean } => {
      let values: number[] | string[];
      
      switch (colorBy) {
        case 'pseudotime':
          values = simulation.metadata.pseudotime;
          break;
        case 'cell_types':
          values = simulation.metadata.cell_types;
          break;
        case 'branch_id':
          values = simulation.metadata.branch_id || [];
          break;
        case 'cycle_phase':
          values = simulation.metadata.cycle_phase || [];
          break;
        case 'cluster_labels':
          values = simulation.metadata.cluster_labels || [];
          break;
        default:
          values = simulation.metadata.pseudotime;
      }

      const isNumeric = typeof values[0] === 'number';
      return { values, isNumeric };
    };

    const { values: colorValues, isNumeric } = getColorValues();

    // ✅ FIX: Handle categorical (string) colors properly
    let markerColor: any;
    let markerColorscale: any;
    let showscale: boolean;
    let colorbar: any;

    if (isNumeric) {
      // Numeric colors (continuous scale)
      markerColor = colorValues;
      markerColorscale = 'Viridis';
      showscale = true;
      colorbar = {
        title: { text: colorBy.replace('_', ' ') },
        thickness: 15,
      };
    } else {
      // Categorical colors (discrete)
      // Convert strings to numeric indices for coloring
      const uniqueValues = Array.from(new Set(colorValues as string[]));
      const colorMap = new Map(uniqueValues.map((v, i) => [v, i]));
      markerColor = (colorValues as string[]).map(v => colorMap.get(v));
      
      // Use a qualitative colorscale for categories
      markerColorscale = 'Portland'; // Good for discrete categories
      showscale = true;
      colorbar = {
        title: { text: colorBy.replace('_', ' ') },
        thickness: 15,
        tickmode: 'array',
        tickvals: Array.from({ length: uniqueValues.length }, (_, i) => i),
        ticktext: uniqueValues,
      };
    }

    const trace = {
      x,
      y,
      mode: 'markers' as const,
      type: 'scatter' as const,
      marker: {
        size: 5,
        color: markerColor,
        colorscale: markerColorscale,
        showscale: showscale,
        colorbar: colorbar,
        line: { width: 0.5, color: 'rgba(0,0,0,0.1)' },
      },
      text: colorValues.map((v, i) => `Cell ${i}: ${v}`),
      hovertemplate: '<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
    };

    const plotLayout = {
      title: {
        text: `${embeddingMethod.toUpperCase()} - ${simulation.parameters.trajectory_type} (continuity: ${simulation.parameters.continuity.toFixed(2)})`,
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
      height: 600,
      hovermode: 'closest' as const,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 60, r: 40, t: 80, b: 60 },
      transition: {
        duration: 500,
        easing: 'cubic-in-out'
      },
    };

    return { data: [trace], layout: plotLayout };
  }, [embedding, embeddingMethod, colorBy, simulation]);

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'] as any,
    // Enable smooth animations
    animation: {
      duration: 500,
      easing: 'cubic-in-out'
    },
  };

  if (!embedding || embedding.length === 0) {
    return (
      <div className="bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg p-8">
        <div className="flex items-center justify-center min-h-[400px] text-[var(--color-muted-foreground)]">
          <p>No embedding data available for {embeddingMethod}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg overflow-hidden">
      <Plot data={data} layout={layout} config={config} style={{ width: '100%' }} />
    </div>
  );
}
