#environment.py

from .model import iVAE
from .mixin import envMixin
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader, TensorDataset


class Env(iVAE, envMixin):
    def __init__(
        self,
        adata,
        layer,
        recon,
        irecon,
        beta,
        dip,
        tc,
        info,
        hidden_dim,
        latent_dim,
        i_dim,
        use_ode,
        loss_mode,
        lr,
        vae_reg,
        ode_reg,
        device,
        train_size=0.7,        # NEW: train split ratio
        val_size=0.15,         # NEW: validation split ratio
        test_size=0.15,        # NEW: test split ratio
        batch_size=128,        # NEW: explicit batch size
        random_seed=42,        # NEW: for reproducibility
        *args,
        **kwargs,
    ):
        # NEW: Store split parameters
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size_fixed = batch_size  # Renamed to avoid confusion
        self.random_seed = random_seed
        
        # Register data with splits
        self._register_anndata(adata, layer, latent_dim, loss_mode)
        
        super().__init__(
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            state_dim=self.n_var,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            device=device,
        )
        
        # NEW: Initialize tracking
        self.score = []
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        
        # NEW: Early stopping parameters
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

    def _register_anndata(self, adata, layer: str, latent_dim: int, loss_mode: str):
        """Register AnnData and create train/val/test splits"""
        
        # Load data based on layer specification
        if layer == 'X':
            data = adata.X
        else:
            if layer not in adata.layers:
                raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")
            data = adata.layers[layer]
        
        # Convert to dense array if sparse
        if hasattr(data, 'toarray'):
            data = data.toarray()
        
        # Check if data appears to be raw counts
        max_val = data.max()
        has_decimals = np.any((data % 1) != 0)
        
        if has_decimals or max_val < 10:
            print(f"\n⚠️  Warning: Layer '{layer}' may not contain raw counts.")
            print(f"   Max value: {max_val:.2f}, Contains decimals: {has_decimals}")
            print(f"   Consider using raw count data for better results.\n")
        
        # Apply log1p transformation
        self.X = np.log1p(data) if loss_mode == 'mse' else data
        
        self.n_obs = adata.shape[0]
        self.n_var = adata.shape[1]
        
        # Generate labels for evaluation
        if 'cell_type' in adata.obs.columns:
            from sklearn.preprocessing import LabelEncoder
            self.labels = LabelEncoder().fit_transform(adata.obs['cell_type'])
        else:
            self.labels = KMeans(latent_dim, random_state=self.random_seed).fit_predict(self.X)
        
        # Create train/val/test splits
        np.random.seed(self.random_seed)
        indices = np.random.permutation(self.n_obs)
        
        n_train = int(self.train_size * self.n_obs)
        n_val = int(self.val_size * self.n_obs)
        
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]
        
        # Split data
        self.X_train = self.X[self.train_idx]
        self.X_val = self.X[self.val_idx]
        self.X_test = self.X[self.test_idx]
        
        self.labels_train = self.labels[self.train_idx]
        self.labels_val = self.labels[self.val_idx]
        self.labels_test = self.labels[self.test_idx]
        
        print("\nData split:")
        print(f"  Train: {len(self.train_idx):,} cells ({len(self.train_idx)/self.n_obs*100:.1f}%)")
        print(f"  Val:   {len(self.val_idx):,} cells ({len(self.val_idx)/self.n_obs*100:.1f}%)")
        print(f"  Test:  {len(self.test_idx):,} cells ({len(self.test_idx)/self.n_obs*100:.1f}%)")
        
        # Create PyTorch DataLoaders
        self._create_dataloaders()
        
        return

    # NEW: Create DataLoaders
    def _create_dataloaders(self):
        """Create PyTorch DataLoaders for train/val/test sets"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        X_val_tensor = torch.FloatTensor(self.X_val)
        X_test_tensor = torch.FloatTensor(self.X_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor)
        val_dataset = TensorDataset(X_val_tensor)
        test_dataset = TensorDataset(X_test_tensor)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size_fixed,
            shuffle=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size_fixed,
            shuffle=False,
            drop_last=False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size_fixed,
            shuffle=False,
            drop_last=False
        )
        
        print(f"  Batches per epoch: {len(self.train_loader)}")

    # NEW: Training for one full epoch
    def train_epoch(self):
        """Train for one complete epoch through training data"""
        
        self.train()  # Set model to training mode
        epoch_losses = []
        
        for batch_data, in self.train_loader:
            batch_data = batch_data.to(self.device)
            self.update(batch_data)
            epoch_losses.append(self.loss[-1][0])  # Get last loss
        
        avg_train_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_train_loss)
        
        return avg_train_loss

    # NEW: Validation evaluation
    def validate(self):
        """Evaluate on validation set"""
        
        self.eval()  # Set model to evaluation mode
        val_losses = []
        all_latents = []
        
        with torch.no_grad():
            for batch_data, in self.val_loader:
                batch_data = batch_data.to(self.device)
                
                # Forward pass (compute loss without updating)
                loss_value = self._compute_loss_only(batch_data)
                val_losses.append(loss_value)
                
                # Get latent representations
                latent = self.take_latent(batch_data)
                all_latents.append(latent)
        
        # Average validation loss
        avg_val_loss = np.mean(val_losses)
        self.val_losses.append(avg_val_loss)
        
        # Compute metrics on validation latents
        all_latents = np.concatenate(all_latents, axis=0)
        val_score = self._calc_score_with_labels(all_latents, self.labels_val)
        self.val_scores.append(val_score)
        
        return avg_val_loss, val_score

    # NEW: Check early stopping
    def check_early_stopping(self, val_loss, patience=20):
        """
        Check if training should stop early
        
        Args:
            val_loss: Current validation loss
            patience: Number of epochs to wait before stopping
            
        Returns:
            should_stop: Boolean indicating if training should stop
        """
        
        if val_loss < self.best_val_loss:
            # Improvement
            self.best_val_loss = val_loss
            self.best_model_state = {
                k: v.cpu().clone() for k, v in self.state_dict().items()
            }
            self.patience_counter = 0
            return False, True  # Continue training, improved
        else:
            # No improvement
            self.patience_counter += 1
            
            if self.patience_counter >= patience:
                return True, False  # Stop training, not improved
            else:
                return False, False  # Continue training, not improved

    # NEW: Load best model
    def load_best_model(self):
        """Load the best model from early stopping checkpoint"""
        if self.best_model_state is not None:
            self.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with validation loss: {self.best_val_loss:.4f}")
        else:
            print("\nWarning: No best model state found!")


        