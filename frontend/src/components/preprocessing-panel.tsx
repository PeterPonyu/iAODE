// ============================================================================
// components/preprocessing-panel.tsx - REFINED
// ============================================================================

'use client';

import { useState } from 'react';
import { 
  applyTFIDF, 
  selectHVP, 
  subsampleData,
  DEFAULT_TFIDF_PARAMS,
  DEFAULT_HVP_PARAMS,
  DEFAULT_SUBSAMPLE_PARAMS
} from '@/lib/api';
import {
  TFIDFParams,
  HVPParams,
  SubsampleParams,
  PreprocessInfo
} from '@/lib/types';

type PreprocessingStep = 'idle' | 'tfidf' | 'hvp' | 'complete';

interface PreprocessingPanelProps {
  dataType: 'scrna' | 'scatac';
  onComplete?: (info: PreprocessInfo) => void;
}

export function PreprocessingPanel({ dataType, onComplete }: PreprocessingPanelProps) {
  const [currentStep, setCurrentStep] = useState<PreprocessingStep>('idle');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preprocessInfo, setPreprocessInfo] = useState<PreprocessInfo | null>(null);

  // TF-IDF params
  const [tfidfParams, setTfidfParams] = useState<TFIDFParams>(DEFAULT_TFIDF_PARAMS);
  
  // HVP params
  const [hvpParams, setHvpParams] = useState<HVPParams>(DEFAULT_HVP_PARAMS);
  
  // Subsample params
  const [subsampleParams, setSubsampleParams] = useState<SubsampleParams>(DEFAULT_SUBSAMPLE_PARAMS);
  const [useCellCount, setUseCellCount] = useState(true);

  // Only show for scATAC data
  if (dataType !== 'scatac') {
    return null;
  }

  const handleTFIDF = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      const result = await applyTFIDF(tfidfParams);
      setPreprocessInfo(result);
      setCurrentStep('tfidf');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'TF-IDF normalization failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleHVP = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      const result = await selectHVP(hvpParams);
      setPreprocessInfo(result);
      setCurrentStep('hvp');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Peak selection failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSubsample = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      const params = useCellCount 
        ? { ...subsampleParams, frac_cells: undefined }
        : { ...subsampleParams, n_cells: undefined };
      
      const result = await subsampleData(params);
      setPreprocessInfo(result);
      setCurrentStep('complete');
      onComplete?.(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Subsampling failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSkipSubsample = () => {
    setCurrentStep('complete');
    if (preprocessInfo) {
      onComplete?.(preprocessInfo);
    }
  };

  return (
    <div className="card">
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2 text-[rgb(var(--text-primary))]">scATAC-seq Preprocessing</h2>
        <p className="text-sm text-[rgb(var(--muted-foreground))]">
          Sequential preprocessing pipeline required for scATAC-seq data
        </p>
      </div>

      {error && (
        <div className="mb-6 p-3 rounded-lg badge-error">
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}

      {preprocessInfo && (
        <div className="mb-6 p-4 rounded-lg card-inner">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-[rgb(var(--muted-foreground))] mb-1">Current Dataset</p>
              <p className="text-lg font-semibold text-[rgb(var(--text-primary))]">
                {preprocessInfo.n_cells.toLocaleString()} cells × {preprocessInfo.n_peaks.toLocaleString()} peaks
              </p>
              <p className="text-sm text-[rgb(var(--muted-foreground))] mt-1">{preprocessInfo.message}</p>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-6">
        {/* Step 1: TF-IDF Normalization */}
        <div className="pb-6 border-t border-[rgb(var(--border))] pt-6 first:border-t-0 first:pt-0">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-xs font-semibold badge-blue px-2.5 py-1 rounded-full">1</span>
            <h3 className="font-semibold text-[rgb(var(--text-primary))]">TF-IDF Normalization</h3>
            {currentStep !== 'idle' && (
              <span className="text-xs badge-success px-2.5 py-1 rounded-full ml-auto">✓ Complete</span>
            )}
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">Scale Factor</label>
                <select
                  value={tfidfParams.scale_factor}
                  onChange={(e) => setTfidfParams({ ...tfidfParams, scale_factor: Number(e.target.value) })}
                  disabled={currentStep !== 'idle' || isProcessing}
                  className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
                >
                  <option value={1e4}>10,000 (Standard)</option>
                  <option value={1e6}>1,000,000 (Large datasets)</option>
                </select>
                <p className="text-xs text-[rgb(var(--muted-foreground))] mt-1">Normalization scale factor</p>
              </div>

              <div className="flex flex-col justify-end space-y-3">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={tfidfParams.log_tf}
                    onChange={(e) => setTfidfParams({ ...tfidfParams, log_tf: e.target.checked })}
                    disabled={currentStep !== 'idle' || isProcessing}
                    className="rounded border border-[rgb(var(--border))]"
                  />
                  <span className="text-[rgb(var(--text-secondary))]">Log-transform TF</span>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={tfidfParams.log_idf}
                    onChange={(e) => setTfidfParams({ ...tfidfParams, log_idf: e.target.checked })}
                    disabled={currentStep !== 'idle' || isProcessing}
                    className="rounded border border-[rgb(var(--border))]"
                  />
                  <span className="text-[rgb(var(--text-secondary))]">Log-transform IDF (Recommended)</span>
                </label>
              </div>
            </div>

            <button
              onClick={handleTFIDF}
              disabled={currentStep !== 'idle' || isProcessing}
              className="btn-primary px-4 py-2 rounded-lg text-sm font-medium"
            >
              {isProcessing && currentStep === 'idle' ? 'Processing...' : 'Apply TF-IDF Normalization'}
            </button>
          </div>
        </div>

        {/* Step 2: Select Highly Variable Peaks */}
        <div className="pb-6 border-t border-[rgb(var(--border))] pt-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-xs font-semibold badge-green px-2.5 py-1 rounded-full">2</span>
            <h3 className="font-semibold text-[rgb(var(--text-primary))]">Select Highly Variable Peaks</h3>
            {(currentStep === 'hvp' || currentStep === 'complete') && (
              <span className="text-xs badge-success px-2.5 py-1 rounded-full ml-auto">✓ Complete</span>
            )}
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">Number of Peaks</label>
                <input
                  type="number"
                  value={hvpParams.n_top_peaks}
                  onChange={(e) => setHvpParams({ ...hvpParams, n_top_peaks: Number(e.target.value) })}
                  disabled={currentStep !== 'tfidf' || isProcessing}
                  className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
                  min={1000}
                  max={100000}
                  step={1000}
                />
                <p className="text-xs text-[rgb(var(--muted-foreground))] mt-1">Top variable peaks to select</p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">Selection Method</label>
                <select
                  value={hvpParams.method}
                  onChange={(e) => setHvpParams({ ...hvpParams, method: e.target.value as 'signac' | 'snapatac2' | 'deviance' })}
                  disabled={currentStep !== 'tfidf' || isProcessing}
                  className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
                >
                  <option value="signac">Signac (Recommended)</option>
                  <option value="snapatac2">SnapATAC2</option>
                  <option value="deviance">Binomial Deviance</option>
                </select>
                <p className="text-xs text-[rgb(var(--muted-foreground))] mt-1">Peak selection algorithm</p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">Min Accessibility</label>
                <input
                  type="number"
                  value={hvpParams.min_accessibility}
                  onChange={(e) => setHvpParams({ ...hvpParams, min_accessibility: Number(e.target.value) })}
                  disabled={currentStep !== 'tfidf' || isProcessing}
                  className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
                  min={0}
                  max={1}
                  step={0.01}
                />
                <p className="text-xs text-[rgb(var(--muted-foreground))] mt-1">Filter peaks in &lt; X% cells</p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">Max Accessibility</label>
                <input
                  type="number"
                  value={hvpParams.max_accessibility}
                  onChange={(e) => setHvpParams({ ...hvpParams, max_accessibility: Number(e.target.value) })}
                  disabled={currentStep !== 'tfidf' || isProcessing}
                  className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
                  min={0}
                  max={1}
                  step={0.01}
                />
                <p className="text-xs text-[rgb(var(--muted-foreground))] mt-1">Filter ubiquitous peaks in &gt; X% cells</p>
              </div>
            </div>

            <button
              onClick={handleHVP}
              disabled={currentStep !== 'tfidf' || isProcessing}
              className="btn-primary px-4 py-2 rounded-lg text-sm font-medium"
            >
              {isProcessing && currentStep === 'tfidf' ? 'Processing...' : 'Select Variable Peaks'}
            </button>
          </div>
        </div>

        {/* Step 3: Subsample (Optional) */}
        <div className="pb-6 border-t border-[rgb(var(--border))] pt-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-xs font-semibold badge-purple px-2.5 py-1 rounded-full">3</span>
            <h3 className="font-semibold text-[rgb(var(--text-primary))]">Subsample Data</h3>
            <span className="text-xs text-[rgb(var(--muted-foreground))]">(Optional)</span>
            {currentStep === 'complete' && (
              <span className="text-xs badge-success px-2.5 py-1 rounded-full ml-auto">✓ Complete</span>
            )}
          </div>

          <div className="space-y-4">
            <div className="flex gap-4 p-3 rounded-lg card-inner">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="radio"
                  checked={useCellCount}
                  onChange={() => setUseCellCount(true)}
                  disabled={currentStep !== 'hvp' || isProcessing}
                  className="text-[rgb(var(--primary))]"
                />
                <span className="text-[rgb(var(--text-secondary))]">Sample by cell count</span>
              </label>
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="radio"
                  checked={!useCellCount}
                  onChange={() => setUseCellCount(false)}
                  disabled={currentStep !== 'hvp' || isProcessing}
                  className="text-[rgb(var(--primary))]"
                />
                <span className="text-[rgb(var(--text-secondary))]">Sample by fraction</span>
              </label>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {useCellCount ? (
                <div>
                  <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">Number of Cells</label>
                  <input
                    type="number"
                    value={subsampleParams.n_cells || ''}
                    onChange={(e) => setSubsampleParams({ ...subsampleParams, n_cells: Number(e.target.value) })}
                    disabled={currentStep !== 'hvp' || isProcessing}
                    className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
                    min={100}
                    step={100}
                    placeholder="e.g., 10000"
                  />
                  <p className="text-xs text-[rgb(var(--muted-foreground))] mt-1">Number of cells to randomly sample</p>
                </div>
              ) : (
                <div>
                  <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">Fraction (0-1)</label>
                  <input
                    type="number"
                    value={subsampleParams.frac_cells || ''}
                    onChange={(e) => setSubsampleParams({ ...subsampleParams, frac_cells: Number(e.target.value) })}
                    disabled={currentStep !== 'hvp' || isProcessing}
                    className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
                    min={0.01}
                    max={1}
                    step={0.1}
                    placeholder="e.g., 0.5"
                  />
                  <p className="text-xs text-[rgb(var(--muted-foreground))] mt-1">Fraction of total cells to sample</p>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium mb-2 text-[rgb(var(--muted-foreground))]">Random Seed</label>
                <input
                  type="number"
                  value={subsampleParams.seed}
                  onChange={(e) => setSubsampleParams({ ...subsampleParams, seed: Number(e.target.value) })}
                  disabled={currentStep !== 'hvp' || isProcessing}
                  className="w-full px-3 py-2 rounded-lg border border-[rgb(var(--border))] text-sm bg-[rgb(var(--background))]"
                />
                <p className="text-xs text-[rgb(var(--muted-foreground))] mt-1">For reproducible sampling</p>
              </div>
            </div>

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={subsampleParams.use_hvp}
                onChange={(e) => setSubsampleParams({ ...subsampleParams, use_hvp: e.target.checked })}
                disabled={currentStep !== 'hvp' || isProcessing}
                className="rounded border border-[rgb(var(--border))]"
              />
              <span className="text-[rgb(var(--text-secondary))]">Filter to highly variable peaks only</span>
            </label>

            <div className="flex gap-3">
              <button
                onClick={handleSubsample}
                disabled={currentStep !== 'hvp' || isProcessing}
                className="btn-primary px-4 py-2 rounded-lg text-sm font-medium"
              >
                {isProcessing && currentStep === 'hvp' ? 'Processing...' : 'Apply Subsampling'}
              </button>

              <button
                onClick={handleSkipSubsample}
                disabled={currentStep !== 'hvp' || isProcessing}
                className="btn-secondary px-4 py-2 rounded-lg text-sm font-medium"
              >
                Skip Subsampling
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
