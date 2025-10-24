/**
 * Main parameter control panel
 */

'use client';

import { useState } from 'react';
import { TrajectoryType } from '@/types/simulation';
import { Card, CardHeader, CardTitle, CardContent, CardFooter, Button } from '@/components/ui';
import { TrajectorySelector } from './TrajectorySelector';
import { ContinuitySlider } from './ContinuitySlider';

export interface ParameterPanelProps {
  onLoad: (trajectoryType: TrajectoryType, continuity: number) => void;
  isLoading?: boolean;
  className?: string;
}

export function ParameterPanel({
  onLoad,
  isLoading = false,
  className = '',
}: ParameterPanelProps) {
  const [trajectoryType, setTrajectoryType] = useState<TrajectoryType>('linear');
  const [continuity, setContinuity] = useState(0.95);

  const handleLoad = () => {
    onLoad(trajectoryType, continuity);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Simulation Parameters</CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        <TrajectorySelector
          value={trajectoryType}
          onChange={setTrajectoryType}
          disabled={isLoading}
        />
        
        <ContinuitySlider
          value={continuity}
          onChange={setContinuity}
          disabled={isLoading}
        />

        {/* Parameter Summary */}
        <div className="p-3 rounded bg-muted">
          <p className="text-sm font-semibold mb-2">Current Selection:</p>
          <div className="text-xs space-y-1 text-muted-foreground">
            <p>• Trajectory: <strong>{trajectoryType}</strong></p>
            <p>• Continuity: <strong>{(continuity * 100).toFixed(0)}%</strong></p>
            <p>• Dataset ID: <strong>{trajectoryType}_{Math.round(continuity * 100)}</strong></p>
          </div>
        </div>
      </CardContent>
      
      <CardFooter>
        <Button
          onClick={handleLoad}
          isLoading={isLoading}
          fullWidth
          variant="primary"
        >
          {isLoading ? 'Loading...' : 'Load Simulation'}
        </Button>
      </CardFooter>
    </Card>
  );
}