#agent.py 

from .environment import Env
from anndata import AnnData
import torch
import tqdm
import time
from typing import Literal, Optional


class agent(Env):
    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
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
        train_size: float = 0.7,      
        val_size: float = 0.15,       
        test_size: float = 0.15,      
        batch_size: int = 128,        
        random_seed: int = 42,
        encoder_type: Literal["mlp", "mlp_residual", "linear", "transformer"] = "mlp",
        encoder_num_layers: int = 2,
        encoder_n_heads: int = 4,
        encoder_d_model: Optional[int] = None,                
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        super().__init__(
            adata=adata,
            layer=layer,
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
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            random_seed=random_seed,
            device=device,
            encoder_type=encoder_type,
            encoder_num_layers=encoder_num_layers,
            encoder_n_heads=encoder_n_heads,
            encoder_d_model=encoder_d_model,
        )
        
        # Initialize resource tracking attributes
        self.train_time = 0.0
        self.peak_memory_gb = 0.0
        self.actual_epochs = 0

    # MODIFIED: Epoch-based training with validation and early stopping
    def fit(
        self, 
        epochs: int = 100,
        patience: int = 20,
        val_every: int = 5,
        early_stop: bool = True,
    ):
        """
        Train model with epoch-based training and early stopping
        
        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience (epochs without improvement)
            val_every: Validate every N epochs
            early_stop: Whether to use early stopping
        """
        
        # Reset GPU memory stats and start timer
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with tqdm.tqdm(total=epochs, desc="Training", ncols=225) as pbar:
            for epoch in range(epochs):
                
                # Train for one epoch
                train_loss = self.train_epoch()
                
                # Validate periodically
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    val_loss, val_score = self.validate()
                    if early_stop:
                        # Check early stopping
                        should_stop, improved = self.check_early_stopping(
                            val_loss, patience
                        )
                        
                        # Update progress bar
                        pbar.set_postfix({
                            "Train": f"{train_loss:.2f}",
                            "Val": f"{val_loss:.2f}",
                            "ARI": f"{val_score[0]:.2f}",
                            "NMI": f"{val_score[1]:.2f}",
                            "ASW": f"{val_score[2]:.2f}",
                            "CAL": f"{val_score[3]:.2f}",
                            "DAV": f"{val_score[4]:.2f}",
                            "COR": f"{val_score[5]:.2f}",
                            "Best": f"{self.best_val_loss:.2f}",
                            "Pat": f"{self.patience_counter}/{patience}",
                            "Imp": "✓" if improved else "✗"
                        })
                        
                        if should_stop:
                            self.actual_epochs = epoch + 1
                            print(f"\n\nEarly stopping triggered at epoch {epoch + 1}")
                            print(f"Best validation loss: {self.best_val_loss:.4f}")
                            break
                    else:
                        # Update progress bar without early stopping
                        pbar.set_postfix({
                            "Train": f"{train_loss:.2f}",
                            "Val": f"{val_loss:.2f}",
                            "ARI": f"{val_score[0]:.2f}",
                            "NMI": f"{val_score[1]:.2f}",
                            "ASW": f"{val_score[2]:.2f}",
                            "CAL": f"{val_score[3]:.2f}",
                            "DAV": f"{val_score[4]:.2f}",
                            "COR": f"{val_score[5]:.2f}",
                        })
                pbar.update(1)
            else:
                # Loop completed without early stopping
                self.actual_epochs = epochs
        
        # Record time and GPU memory
        self.train_time = time.time() - start_time
        self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
        
        # Print resource usage summary
        print(f"\n{'='*70}")
        print(f"Training completed")
        print(f"  Epochs: {self.actual_epochs}/{epochs}")
        print(f"  Time: {self.train_time:.2f}s")
        print(f"  Peak GPU Memory: {self.peak_memory_gb:.3f} GB")
        print(f"{'='*70}\n")
        
        return self

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
        return self.take_latent(torch.FloatTensor(self.X_test).to(self.device))
    
    def get_resource_metrics(self):
        """
        Get resource usage metrics from training
        """
        return {
            'train_time': self.train_time,
            'peak_memory_gb': self.peak_memory_gb,
            'actual_epochs': self.actual_epochs
        }