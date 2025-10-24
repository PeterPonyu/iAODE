/**
 * Embedding method selector
 */

'use client';

import { SimulationData, EmbeddingArray } from '@/types/simulation';
import { Select, SelectOption, Badge } from '@/components/ui';

export interface EmbeddingSelectorProps {
  simulation: SimulationData | null;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
}

export function EmbeddingSelector({
  simulation,
  value,
  onChange,
  disabled = false,
  className = '',
}: EmbeddingSelectorProps) {
  if (!simulation) {
    return (
      <div className={className}>
        <label className="block text-sm font-medium mb-2">
          Embedding Method
        </label>
        <div className="p-3 rounded bg-muted border">
          <p className="text-sm text-muted-foreground">
            No simulation loaded
          </p>
        </div>
      </div>
    );
  }

  const embeddingOptions: SelectOption[] = Object.keys(simulation.embeddings)
    .filter((key) => simulation.embeddings[key] !== undefined)
    .map((key) => ({
      value: key,
      label: key.toUpperCase(),
    }));

  if (embeddingOptions.length === 0) {
    return (
      <div className={className}>
        <label className="block text-sm font-medium mb-2">
          Embedding Method
        </label>
        <div className="p-3 rounded bg-muted border">
          <p className="text-sm text-muted-foreground">
            No embeddings available
          </p>
        </div>
      </div>
    );
  }

  // Safely get the current embedding with type assertion
  const currentEmbedding: EmbeddingArray | undefined = value 
    ? (simulation.embeddings[value] as EmbeddingArray | undefined)
    : undefined;

  return (
    <div className={className}>
      <Select
        label="Embedding Method"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        options={embeddingOptions}
        disabled={disabled}
        helperText={`${embeddingOptions.length} embedding(s) available`}
      />
      
      {/* Embedding Info */}
      {currentEmbedding && currentEmbedding.length > 0 && (
        <div className="mt-3 p-3 rounded bg-muted space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Dimensions:</span>
            <Badge size="sm">
              {currentEmbedding[0]?.length ?? 2}D
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Data Points:</span>
            <Badge size="sm">{currentEmbedding.length}</Badge>
          </div>
        </div>
      )}
    </div>
  );
}