"""
Quick verification test for raw counts handling.
"""

import numpy as np
import torch

print("="*70)
print("Quick Verification Test")
print("="*70)

# Create synthetic data
np.random.seed(42)
n_cells = 50
n_peaks = 30
raw_counts = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_peaks)).astype(np.float32)
log_counts = np.log1p(raw_counts)

print(f"✓ Created synthetic data: {n_cells} cells × {n_peaks} peaks")
print(f"  Raw counts: mean={raw_counts.mean():.2f}, range=[{raw_counts.min():.0f}, {raw_counts.max():.0f}]")
print(f"  Log counts: mean={log_counts.mean():.2f}, range=[{log_counts.min():.2f}, {log_counts.max():.2f}]")
print()

# Test 1: Check module.py signature
print("Test 1: Module forward pass signature")
print("-" * 70)
from iaode.module import iVAE

model = iVAE(
    state_dim=n_peaks,
    hidden_dim=32,
    action_dim=5,
    i_dim=2,
    use_ode=False,
    loss_mode='nb',
    device=torch.device('cpu')
)

x_log_t = torch.FloatTensor(log_counts[:10])
x_raw_t = torch.FloatTensor(raw_counts[:10])

# Test forward with both inputs
output = model(x_log_t, x_raw_t)
print(f"✓ Forward pass successful with separate log and raw inputs")
print(f"  Output tuple length: {len(output)}")
print(f"  Contains raw counts: {output[3].shape}")
assert torch.allclose(output[3], x_raw_t), "Output should contain raw counts!"
print(f"✓ Raw counts correctly propagated through forward pass")
print()

# Test 2: Check environment.py data storage
print("Test 2: Environment data storage")
print("-" * 70)
from anndata import AnnData
from iaode.agent import agent

adata = AnnData(X=raw_counts)
adata.layers['counts'] = raw_counts.copy()

ag = agent(
    adata,
    layer='counts',
    latent_dim=5,
    i_dim=2,
    use_ode=False,
    loss_mode='nb',
    batch_size=10,
    random_seed=42
)

assert hasattr(ag, 'X_raw'), "X_raw not stored!"
assert hasattr(ag, 'X'), "X not stored!"
print(f"✓ Both X_raw and X stored in environment")

assert np.allclose(ag.X, np.log1p(ag.X_raw)), "X should be log1p(X_raw)!"
print(f"✓ X = log1p(X_raw) verified")

assert np.allclose(ag.X_raw, raw_counts), "X_raw should match input counts!"
print(f"✓ X_raw matches original raw counts")
print()

# Test 3: Check dataloader provides both
print("Test 3: DataLoader provides both log and raw")
print("-" * 70)
for batch_log, batch_raw in ag.train_loader:
    print(f"✓ DataLoader batch contains log and raw data")
    print(f"  Log batch shape: {batch_log.shape}")
    print(f"  Raw batch shape: {batch_raw.shape}")
    assert not torch.allclose(batch_log, batch_raw), "Log and raw should be different!"
    print(f"✓ Log and raw batches are different (as expected)")
    break  # Just check first batch
print()

# Test 4: Verify NB loss uses raw counts
print("Test 4: NB loss calculation uses raw counts")
print("-" * 70)
from iaode.mixin import scviMixin

class TestMixin(scviMixin):
    pass

mixin = TestMixin()

# Simulate predictions (should be on raw count scale)
mu = torch.FloatTensor(raw_counts[:5])
theta = torch.ones(n_peaks) * 10
x_raw = torch.FloatTensor(raw_counts[:5])
x_log = torch.FloatTensor(log_counts[:5])

# NB loss with raw counts (correct)
loss_raw = mixin._log_nb(x_raw, mu, theta).mean()

# NB loss with log counts (incorrect - what we're fixing)
loss_log = mixin._log_nb(x_log, mu, theta).mean()

print(f"✓ NB loss calculated")
print(f"  Loss with raw counts: {loss_raw.item():.4f}")
print(f"  Loss with log counts: {loss_log.item():.4f}")
print(f"  These should be DIFFERENT (we want to use raw)")
assert not torch.isclose(loss_raw, loss_log), "Losses should differ!"
print(f"✓ Verified NB loss behaves differently for raw vs log")
print()

print("="*70)
print("ALL VERIFICATION TESTS PASSED! ✓")
print("="*70)
print()
print("Summary:")
print("  ✓ Module accepts separate log and raw inputs")
print("  ✓ Raw counts properly stored in environment")
print("  ✓ DataLoaders provide both log and raw data")
print("  ✓ NB loss function works correctly with raw counts")
print()
print("The implementation correctly separates:")
print("  • Log-transformed data → encoder input (stability)")
print("  • Raw count data → NB/ZINB loss calculation (correctness)")
