#agent.py 

from .environment import Env
from .utils import quiver_autoscale, l2_norm
import scanpy as sc
from anndata import AnnData
import torch
import tqdm
from typing import Optional, Literal
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
import pandas as pd


class agent(Env):
    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
        percent: float = 0.01,  # DEPRECATED: kept for compatibility
        recon: float = 1.0,
        irecon: float = 0.0,
        beta: float = 1.0,
        dip: float = 0.0,
        tc: float = 0.0,
        info: float = 0.0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        use_ode: bool = False,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        lr: float = 1e-4,
        vae_reg: float = 0.5,
        ode_reg: float = 0.5,
        train_size: float = 0.7,      # NEW
        val_size: float = 0.15,       # NEW
        test_size: float = 0.15,      # NEW
        batch_size: int = 128,        # NEW
        random_seed: int = 42,        # NEW
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        super().__init__(
            adata=adata,
            layer=layer,
            percent=percent,
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            train_size=train_size,     # NEW
            val_size=val_size,         # NEW
            test_size=test_size,       # NEW
            batch_size=batch_size,     # NEW
            random_seed=random_seed,   # NEW
            device=device,
        )

    # MODIFIED: Epoch-based training with validation and early stopping
    def fit(
        self, 
        epochs: int = 100,
        patience: int = 20,
        val_every: int = 5,
        verbose: bool = True
    ):
        """
        Train model with epoch-based training and early stopping
        
        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience (epochs without improvement)
            val_every: Validate every N epochs
            verbose: Print progress
        """
        
        print(f"\n{'='*70}")
        print(f"Training Configuration")
        print(f"{'='*70}")
        print(f"Max epochs: {epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Validation frequency: every {val_every} epochs")
        print(f"Batch size: {self.batch_size_fixed}")
        print(f"Learning rate: {self.lr}")
        print(f"{'='*70}\n")
        
        with tqdm.tqdm(total=epochs, desc="Training", ncols=150) as pbar:
            for epoch in range(epochs):
                
                # Train for one epoch
                train_loss = self.train_epoch()
                
                # Validate periodically
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    val_loss, val_score = self.validate()
                    
                    # Check early stopping
                    should_stop, improved = self.check_early_stopping(
                        val_loss, patience
                    )
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "Train": f"{train_loss:.3f}",
                        "Val": f"{val_loss:.3f}",
                        "ARI": f"{val_score[0]:.3f}",
                        "NMI": f"{val_score[1]:.3f}",
                        "ASW": f"{val_score[2]:.3f}",
                        "Best": f"{self.best_val_loss:.3f}",
                        "Pat": f"{self.patience_counter}/{patience}",
                        "✓" if improved else "✗": ""
                    })
                    
                    if should_stop:
                        print(f"\n\nEarly stopping triggered at epoch {epoch + 1}")
                        print(f"Best validation loss: {self.best_val_loss:.4f}")
                        break
                else:
                    # Just show training loss
                    pbar.set_postfix({
                        "Train": f"{train_loss:.3f}",
                        "Best": f"{self.best_val_loss:.3f}",
                        "Pat": f"{self.patience_counter}/{patience}"
                    })
                
                pbar.update(1)
        
        # Load best model
        print("\nLoading best model from validation checkpoint...")
        self.load_best_model()
        
        # Final evaluation on test set
        print("\n" + "="*70)
        print("FINAL EVALUATION ON TEST SET")
        print("="*70)
        
        test_loss, test_score, test_latent = self.evaluate_test()
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  ARI:  {test_score[0]:.4f}")
        print(f"  NMI:  {test_score[1]:.4f}")
        print(f"  ASW:  {test_score[2]:.4f}")
        print(f"  C_H:  {test_score[3]:.4f}")
        print(f"  D_B:  {test_score[4]:.4f}")
        print(f"  P_C:  {test_score[5]:.4f}")
        
        # Store test results
        self.test_results = {
            'test_loss': test_loss,
            'test_score': test_score,
            'test_latent': test_latent
        }
        
        return self

    # NEW: Get training history
    def get_training_history(self):
        """Get training history for plotting"""
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores,
        }
        
        return history

    # NEW: Plot training curves
    def plot_training_curves(self, save_path=None):
        """Plot training and validation curves"""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train', alpha=0.7)
        val_epochs = list(range(0, len(self.val_losses) * 5, 5))
        axes[0, 0].plot(val_epochs, self.val_losses, label='Val', marker='o')
        axes[0, 0].axhline(self.best_val_loss, color='r', linestyle='--', 
                          label='Best Val', alpha=0.5)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Metric curves
        if len(self.val_scores) > 0:
            val_scores_array = np.array(self.val_scores)
            metric_names = ['ARI', 'NMI', 'ASW', 'C_H', 'D_B']
            
            for idx, (ax, name) in enumerate(zip(axes.flat[1:6], metric_names)):
                if idx < val_scores_array.shape[1]:
                    ax.plot(val_epochs, val_scores_array[:, idx], marker='o')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(name)
                    ax.set_title(f'Validation {name}')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()

    # NEW: Save results
    def save_results(self, output_dir='./results'):
        """Save training results and test metrics"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training history
        history_df = pd.DataFrame({
            'epoch': list(range(len(self.train_losses))),
            'train_loss': self.train_losses
        })
        history_df.to_csv(f'{output_dir}/training_history.csv', index=False)
        
        # Save validation history
        if len(self.val_losses) > 0:
            val_epochs = list(range(0, len(self.val_losses) * 5, 5))
            val_scores_array = np.array(self.val_scores)
            
            val_df = pd.DataFrame({
                'epoch': val_epochs,
                'val_loss': self.val_losses,
                'val_ARI': val_scores_array[:, 0],
                'val_NMI': val_scores_array[:, 1],
                'val_ASW': val_scores_array[:, 2],
                'val_C_H': val_scores_array[:, 3],
                'val_D_B': val_scores_array[:, 4],
                'val_P_C': val_scores_array[:, 5],
            })
            val_df.to_csv(f'{output_dir}/validation_history.csv', index=False)
        
        # Save test results
        if hasattr(self, 'test_results'):
            test_score = self.test_results['test_score']
            
            test_df = pd.DataFrame({
                'metric': ['Loss', 'ARI', 'NMI', 'ASW', 'C_H', 'D_B', 'P_C'],
                'value': [
                    self.test_results['test_loss'],
                    test_score[0], test_score[1], test_score[2],
                    test_score[3], test_score[4], test_score[5]
                ]
            })
            test_df.to_csv(f'{output_dir}/test_results.csv', index=False)
            
            # Save test latent embeddings
            np.save(f'{output_dir}/test_latent.npy', 
                   self.test_results['test_latent'])
        
        print(f"\nResults saved to {output_dir}/")

    # Keep existing methods
    def get_iembed(self):
        """Get interpretable embedding from full dataset"""
        iembed = self.take_iembed(torch.FloatTensor(self.X).to(self.device))
        return iembed

    def get_latent(self):
        """Get latent representation from full dataset"""
        latent = self.take_latent(torch.FloatTensor(self.X).to(self.device))
        return latent
    
    def get_test_latent(self):
        """Get latent representation from test set only"""
        if hasattr(self, 'test_results'):
            return self.test_results['test_latent']
        else:
            return self.take_latent(torch.FloatTensor(self.X_test).to(self.device))