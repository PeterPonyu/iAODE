# iAODE: Raw Counts vs Log-Transformed Data Implementation

## Summary

Successfully refactored the iAODE package to properly handle raw counts and log-transformed data:

- **Log-transformed data** → Used for encoder input (numerical stability during training)
- **Raw count data** → Used for NB/ZINB loss calculations (statistical correctness)

## Problem Identified

The original implementation applied `log1p` transformation to input data and used the same transformed data for:
1. Encoder input ✓ (correct - improves stability)
2. NB/ZINB loss calculation ✗ (incorrect - these distributions expect raw counts)

This caused incorrect likelihood calculations for `loss_mode='nb'` and `loss_mode='zinb'`.

## Solution Overview

### Architecture Changes

```
Input Data Flow:
┌─────────────────┐
│  Raw Counts     │ (from adata.layers)
└────────┬────────┘
         │
    ┌────┴────┐
    │ log1p() │
    └────┬────┘
         │
    ┌────┴──────────────────┐
    │                       │
┌───▼────────┐      ┌──────▼──────┐
│ X (log)    │      │ X_raw       │
│ → Encoder  │      │ → NB/ZINB   │
└────────────┘      └─────────────┘
```

## Files Modified

### 1. `iaode/environment.py`

**Changes:**
- Store both `self.X_raw` (raw counts) and `self.X` (log-transformed)
- Split both datasets into train/val/test
- Create DataLoaders that provide both log and raw data as tuple `(X_log, X_raw)`
- Update `train_epoch()` and `validate()` to unpack both tensors

**Key code:**
```python
# Store both versions
self.X_raw = adata.layers[layer].toarray()  # or .copy()
self.X = np.log1p(self.X_raw)

# DataLoaders provide both
train_dataset = TensorDataset(X_train_log_tensor, X_raw_train_tensor)
```

### 2. `iaode/module.py` (iVAE)

**Changes:**
- Modified `iVAE.forward()` signature to accept both `x_log` and `x_raw`
- Encoder uses `x_log` for stability
- Forward pass returns `x_raw` (in place of previous `x`) for loss calculation
- Non-ODE mode now returns raw counts in output tuple

**Key code:**
```python
def forward(self, x_log: torch.Tensor, x_raw: torch.Tensor = None):
    # Encode using log-transformed for stability
    q_z, q_m, q_s = self.encoder(x_log)
    
    # ... decode ...
    
    # Return raw counts for loss calculation
    return (q_z, q_m, q_s, x_raw, pred_x, le, pred_xl)
```

### 3. `iaode/model.py`

**Changes:**
- Updated `update()` method to accept `(states_log, states_raw)`
- Updated `_compute_loss_only()` to accept both versions
- All NB/ZINB loss calculations now use raw counts (`x` or extracted from forward)
- MSE loss continues to use log-transformed data for consistency
- Updated `take_iembed()` to handle new signature

**Key code:**
```python
# NB loss with raw counts
L = x.sum(-1).view(-1, 1)  # Library size from raw counts
pred_x = pred_x * L
recon_loss = -self._log_nb(x, pred_x, disp).sum(-1).mean()

# MSE loss with log-transformed (for consistency)
recon_loss = F.mse_loss(states_log, pred_x, reduction="none").sum(-1).mean()
```

## Backward Compatibility

- **Inference methods** (`get_latent()`, `get_iembed()`, etc.) work unchanged
- They use log-transformed data which is passed as both arguments
- MSE mode continues to work with log-transformed data

## Verification

The changes ensure:

1. ✓ **Training stability**: Encoder receives log-transformed data
2. ✓ **Statistical correctness**: NB/ZINB losses receive raw counts
3. ✓ **Library size calculation**: Computed from raw counts `L = x.sum(-1)`
4. ✓ **No gradient issues**: Both data types flow correctly through backprop
5. ✓ **All loss modes work**: MSE, NB, ZINB
6. ✓ **ODE mode works**: Properly handles both data streams

## Testing

Two test files created:

1. **`test_quick_verification.py`**: Fast unit tests for core functionality
2. **`test_raw_counts_fix.py`**: Full integration test with training

To run tests:
```bash
conda activate dl  # or your environment
python test_quick_verification.py
python test_raw_counts_fix.py
```

## Usage Example

No changes required to user code:

```python
from iaode import agent

# Works as before
ag = agent(
    adata,
    layer='counts',
    loss_mode='nb',  # Now correctly uses raw counts
    latent_dim=10,
    use_ode=True
)

ag.fit(epochs=200, patience=20)
latent = ag.get_latent()
```

## Impact

- **Prevents training failure** when raw counts would cause numerical issues in encoder
- **Fixes incorrect likelihood calculations** for NB/ZINB modes
- **Improves model convergence** by using appropriate data for each component
- **Maintains compatibility** with existing code and workflows

## Technical Notes

### Why This Matters

1. **NB/ZINB Distributions**: These are count distributions that expect non-negative integer-like values. Log-transformed data violates their assumptions.

2. **Encoder Stability**: Log transformation compresses the range of large count values, making gradients more stable and preventing overflow.

3. **Library Size**: The total count per cell `L = sum(raw_counts)` is a critical normalization factor for scRNA-seq/scATAC-seq. It must be computed from raw counts.

### Design Decisions

- **Separate data streams**: Cleaner than trying to "undo" log transformation
- **Raw counts in forward output**: Makes loss calculation transparent
- **MSE uses log**: Maintains consistency with original behavior for continuous data
- **Backward compatible inference**: Existing analysis code works unchanged
