// ============================================================================
// components/training-params.tsx - REFINED
// ============================================================================

'use client';

import { useState } from 'react';
import { AgentParams, TrainParams } from '@/lib/types';

type TrainingParamsProps = {
  onSubmit: (agentParams: AgentParams, trainParams: TrainParams) => void;
  disabled: boolean;
};

export function TrainingParams({ onSubmit, disabled }: TrainingParamsProps) {
  const [agentParams, setAgentParams] = useState<AgentParams>({
    layer: 'X',
    recon: 1.0,
    irecon: 0.0,
    beta: 1.0,
    dip: 0.0,
    tc: 0.0,
    info: 0.0,
    hidden_dim: 128,
    latent_dim: 10,
    i_dim: 2,
    use_ode: false,
    loss_mode: 'nb',
    lr: 0.0001,
    vae_reg: 0.5,
    ode_reg: 0.5,
    train_size: 0.7,
    val_size: 0.15,
    test_size: 0.15,
    batch_size: 128,
    random_seed: 42,
  });

  const [trainParams, setTrainParams] = useState<TrainParams>({
    epochs: 100,
    patience: 20,
    val_every: 5,
    early_stop: true,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(agentParams, trainParams);
  };

  return (
    <form onSubmit={handleSubmit} className="card rounded-lg p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Training Configuration</h2>
        <p className="text-sm text-muted">Configure model architecture and training parameters</p>
      </div>

      <div className="space-y-6">
        {/* Model Architecture */}
        <div className="pb-6 border-t pt-6 first:border-t-0 first:pt-0">
          <h3 className="font-semibold mb-4">Model Architecture</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Hidden Dimension</label>
              <input
                type="number"
                value={agentParams.hidden_dim}
                onChange={(e) => setAgentParams({...agentParams, hidden_dim: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={32}
                max={512}
              />
              <p className="text-xs text-muted mt-1">Size of encoder/decoder hidden layers</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Latent Dimension</label>
              <input
                type="number"
                value={agentParams.latent_dim}
                onChange={(e) => setAgentParams({...agentParams, latent_dim: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={2}
                max={100}
              />
              <p className="text-xs text-muted mt-1">Latent embedding dimension</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Interpretable Dimension</label>
              <input
                type="number"
                value={agentParams.i_dim}
                onChange={(e) => setAgentParams({...agentParams, i_dim: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={2}
                max={10}
              />
              <p className="text-xs text-muted mt-1">Interpretable embedding dimension</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Data Layer</label>
              <input
                type="text"
                value={agentParams.layer}
                onChange={(e) => setAgentParams({...agentParams, layer: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                placeholder="X or counts"
              />
              <p className="text-xs text-muted mt-1">AnnData layer (X for .X)</p>
            </div>
          </div>
        </div>

        {/* Loss Configuration */}
        <div className="pb-6 border-t pt-6">
          <h3 className="font-semibold mb-4">Loss Configuration</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Loss Mode</label>
              <select
                value={agentParams.loss_mode}
                onChange={(e) => setAgentParams({...agentParams, loss_mode: e.target.value as 'mse' | 'nb' | 'zinb'})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
              >
                <option value="mse">MSE (Mean Squared Error)</option>
                <option value="nb">NB (Negative Binomial)</option>
                <option value="zinb">ZINB (Zero-Inflated NB)</option>
              </select>
              <p className="text-xs text-muted mt-1">Reconstruction loss function</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Reconstruction Weight</label>
              <input
                type="number"
                step="0.1"
                value={agentParams.recon}
                onChange={(e) => setAgentParams({...agentParams, recon: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={0}
              />
              <p className="text-xs text-muted mt-1">Main reconstruction loss weight</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Beta (KL Weight)</label>
              <input
                type="number"
                step="0.1"
                value={agentParams.beta}
                onChange={(e) => setAgentParams({...agentParams, beta: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={0}
              />
              <p className="text-xs text-muted mt-1">Beta-VAE regularization weight</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Interpretable Recon Weight</label>
              <input
                type="number"
                step="0.1"
                value={agentParams.irecon}
                onChange={(e) => setAgentParams({...agentParams, irecon: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={0}
              />
              <p className="text-xs text-muted mt-1">Interpretable reconstruction weight</p>
            </div>
          </div>

          <div className="mt-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={agentParams.use_ode}
                onChange={(e) => setAgentParams({...agentParams, use_ode: e.target.checked})}
                className="rounded border"
              />
              <span className="font-medium">Enable ODE (Ordinary Differential Equations)</span>
            </label>
            <p className="text-xs text-muted mt-1 ml-6">For trajectory modeling and dynamics learning</p>
          </div>
        </div>

        {/* Training Settings */}
        <div className="pb-6 border-t pt-6">
          <h3 className="font-semibold mb-4">Training Settings</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                value={agentParams.lr}
                onChange={(e) => setAgentParams({...agentParams, lr: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={0.00001}
                max={0.01}
              />
              <p className="text-xs text-muted mt-1">Adam optimizer learning rate</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Batch Size</label>
              <input
                type="number"
                value={agentParams.batch_size}
                onChange={(e) => setAgentParams({...agentParams, batch_size: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={16}
                max={1024}
                step={16}
              />
              <p className="text-xs text-muted mt-1">Training batch size</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Max Epochs</label>
              <input
                type="number"
                value={trainParams.epochs}
                onChange={(e) => setTrainParams({...trainParams, epochs: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={10}
                max={1000}
              />
              <p className="text-xs text-muted mt-1">Maximum training epochs</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-muted">Early Stop Patience</label>
              <input
                type="number"
                value={trainParams.patience}
                onChange={(e) => setTrainParams({...trainParams, patience: Number(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border text-sm"
                min={5}
                max={100}
              />
              <p className="text-xs text-muted mt-1">Epochs to wait before stopping</p>
            </div>
          </div>
        </div>
      </div>

      <button
        type="submit"
        disabled={disabled}
        className="w-full px-4 py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed btn-primary mt-6"
      >
        {disabled ? 'Training in Progress...' : 'Start Training'}
      </button>
    </form>
  );
}
