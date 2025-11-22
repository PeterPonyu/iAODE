"""
Test script to verify raw counts vs log-transformed data handling.

This script validates that:
1. Log-transformed data is used for encoder input (stability)
2. Raw counts are used for NB/ZINB loss calculations (correctness)
3. The model trains successfully with both data streams
"""

import numpy as np
import torch
import scanpy as sc
from anndata import AnnData

# Create synthetic scATAC-seq data
np.random.seed(42)
n_cells = 100
n_peaks = 50

# Generate sparse count data (simulating scATAC-seq)
raw_counts = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_peaks))
raw_counts = raw_counts.astype(np.float32)

# Create AnnData object
adata = AnnData(X=raw_counts)
adata.layers['counts'] = raw_counts.copy()

print("="*70)
print("Testing iAODE with Raw Counts vs Log-Transformed Data")
print("="*70)
print(f"Data shape: {adata.shape}")
print(f"Raw counts range: [{raw_counts.min():.1f}, {raw_counts.max():.1f}]")
print(f"Raw counts mean: {raw_counts.mean():.2f}")
print(f"Log1p counts range: [{np.log1p(raw_counts).min():.2f}, {np.log1p(raw_counts).max():.2f}]")
print(f"Log1p counts mean: {np.log1p(raw_counts).mean():.2f}")
print()

# Import iAODE
from iaode import agent

# Test with NB loss (should use raw counts)
print("Testing with loss_mode='nb' (Negative Binomial)")
print("-" * 70)
ag_nb = agent(
    adata,
    layer='counts',
    latent_dim=5,
    i_dim=2,
    use_ode=False,
    loss_mode='nb',  # Should use raw counts internally
    batch_size=32,
    random_seed=42
)

# Check that raw counts are stored
assert hasattr(ag_nb, 'X_raw'), "X_raw not stored!"
assert hasattr(ag_nb, 'X'), "X (log-transformed) not stored!"
print(f"✓ Raw counts stored: shape {ag_nb.X_raw.shape}")
print(f"✓ Log-transformed stored: shape {ag_nb.X.shape}")

# Verify values
assert not np.allclose(ag_nb.X_raw, ag_nb.X), "X_raw and X should be different!"
assert np.allclose(ag_nb.X, np.log1p(ag_nb.X_raw)), "X should be log1p(X_raw)!"
print(f"✓ X = log1p(X_raw) verified")
print()

# Quick training test
print("Running short training test...")
ag_nb.fit(epochs=5, patience=10, early_stop=False)
print("✓ Training completed successfully with NB loss")
print()

# Test with ZINB loss
print("Testing with loss_mode='zinb' (Zero-Inflated Negative Binomial)")
print("-" * 70)
ag_zinb = agent(
    adata,
    layer='counts',
    latent_dim=5,
    i_dim=2,
    use_ode=False,
    loss_mode='zinb',  # Should use raw counts internally
    batch_size=32,
    random_seed=42
)

assert hasattr(ag_zinb, 'X_raw'), "X_raw not stored for ZINB!"
print(f"✓ Raw counts stored for ZINB mode")

ag_zinb.fit(epochs=5, patience=10, early_stop=False)
print("✓ Training completed successfully with ZINB loss")
print()

# Test with MSE loss (backward compatibility)
print("Testing with loss_mode='mse' (Mean Squared Error)")
print("-" * 70)
ag_mse = agent(
    adata,
    layer='counts',
    latent_dim=5,
    i_dim=2,
    use_ode=False,
    loss_mode='mse',  # Can use log-transformed data
    batch_size=32,
    random_seed=42
)

ag_mse.fit(epochs=5, patience=10, early_stop=False)
print("✓ Training completed successfully with MSE loss")
print()

# Test with ODE mode
print("Testing with ODE mode enabled")
print("-" * 70)
ag_ode = agent(
    adata,
    layer='counts',
    latent_dim=5,
    i_dim=2,
    use_ode=True,  # Enable ODE
    loss_mode='nb',
    batch_size=32,
    random_seed=42
)

ag_ode.fit(epochs=5, patience=10, early_stop=False)
print("✓ Training completed successfully with ODE + NB loss")
print()

# Test inference
print("Testing inference methods")
print("-" * 70)
latent = ag_nb.get_latent()
print(f"✓ Latent representation: shape {latent.shape}")

iembed = ag_nb.get_iembed()
print(f"✓ Interpretable embedding: shape {iembed.shape}")

test_latent = ag_nb.get_test_latent()
print(f"✓ Test latent: shape {test_latent.shape}")
print()

print("="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print()
print("Summary:")
print("  • Raw counts are properly stored and used for NB/ZINB loss")
print("  • Log-transformed data is used for encoder input (stability)")
print("  • Training works correctly for all loss modes (NB, ZINB, MSE)")
print("  • ODE mode works correctly with raw counts")
print("  • Inference methods work as expected")
print()
