
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
    <form onSubmit={handleSubmit} className="space-y-6 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h2 className="text-xl font-semibold">Training Configuration</h2>

      {/* Model Architecture */}
      <div className="space-y-4">
        <h3 className="font-medium text-sm text-gray-700 dark:text-gray-300">Model Architecture</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm mb-1">Hidden Dimension</label>
            <input
              type="number"
              value={agentParams.hidden_dim}
              onChange={(e) => setAgentParams({...agentParams, hidden_dim: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Size of hidden layers in encoder/decoder</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Latent Dimension</label>
            <input
              type="number"
              value={agentParams.latent_dim}
              onChange={(e) => setAgentParams({...agentParams, latent_dim: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Dimension of latent space</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Interpretable Dimension</label>
            <input
              type="number"
              value={agentParams.i_dim}
              onChange={(e) => setAgentParams({...agentParams, i_dim: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Dimension of interpretable embedding</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Data Layer</label>
            <input
              type="text"
              value={agentParams.layer}
              onChange={(e) => setAgentParams({...agentParams, layer: e.target.value})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
              placeholder="X or counts"
            />
            <p className="text-xs text-gray-500 mt-1">AnnData layer to use (X for .X)</p>
          </div>
        </div>
      </div>

      {/* Loss Configuration */}
      <div className="space-y-4">
        <h3 className="font-medium text-sm text-gray-700 dark:text-gray-300">Loss Configuration</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm mb-1">Loss Mode</label>
            <select
              value={agentParams.loss_mode}
              onChange={(e) => setAgentParams({...agentParams, loss_mode: e.target.value as 'mse' | 'nb' | 'zinb'})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            >
              <option value="mse">MSE (Mean Squared Error)</option>
              <option value="nb">NB (Negative Binomial)</option>
              <option value="zinb">ZINB (Zero-Inflated NB)</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">Reconstruction loss type</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Reconstruction Weight</label>
            <input
              type="number"
              step="0.1"
              value={agentParams.recon}
              onChange={(e) => setAgentParams({...agentParams, recon: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Weight for reconstruction loss</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Beta (KL Weight)</label>
            <input
              type="number"
              step="0.1"
              value={agentParams.beta}
              onChange={(e) => setAgentParams({...agentParams, beta: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Beta-VAE regularization weight</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Interpretable Recon</label>
            <input
              type="number"
              step="0.1"
              value={agentParams.irecon}
              onChange={(e) => setAgentParams({...agentParams, irecon: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Weight for interpretable reconstruction</p>
          </div>
        </div>
      </div>

      {/* Training Settings */}
      <div className="space-y-4">
        <h3 className="font-medium text-sm text-gray-700 dark:text-gray-300">Training Settings</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm mb-1">Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              value={agentParams.lr}
              onChange={(e) => setAgentParams({...agentParams, lr: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Optimizer learning rate</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Batch Size</label>
            <input
              type="number"
              value={agentParams.batch_size}
              onChange={(e) => setAgentParams({...agentParams, batch_size: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Training batch size</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Max Epochs</label>
            <input
              type="number"
              value={trainParams.epochs}
              onChange={(e) => setTrainParams({...trainParams, epochs: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Maximum training epochs</p>
          </div>

          <div>
            <label className="block text-sm mb-1">Early Stop Patience</label>
            <input
              type="number"
              value={trainParams.patience}
              onChange={(e) => setTrainParams({...trainParams, patience: Number(e.target.value)})}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">Epochs to wait before stopping</p>
          </div>

          <div className="col-span-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={agentParams.use_ode}
                onChange={(e) => setAgentParams({...agentParams, use_ode: e.target.checked})}
                className="rounded"
              />
              <span className="text-sm">Use ODE (Ordinary Differential Equations)</span>
            </label>
            <p className="text-xs text-gray-500 mt-1 ml-6">Enable ODE-based trajectory modeling</p>
          </div>
        </div>
      </div>

      <button
        type="submit"
        disabled={disabled}
        className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors"
      >
        Start Training
      </button>
    </form>
  );
}
