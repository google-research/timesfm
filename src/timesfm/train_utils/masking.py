
from __future__ import annotations
import torch

def sample_first_patch_mask(patch_len: int, batch_size: int, device=None) -> torch.Tensor:
    """
    Training-time random patch masking (per TimesFM paper):
      - For each series in the batch, sample r ~ Uniform{0, 1, ..., patch_len-1}
      - Set m[0:r] = 1 (masked), m[r:] = 0 for the FIRST input patch only
    Returns:
      mask: Bool tensor of shape [batch_size, patch_len], True where positions are masked.
    """
    if patch_len <= 0 or batch_size <= 0:
        raise ValueError("patch_len and batch_size must be positive.")
    r = torch.randint(low=0, high=patch_len, size=(batch_size,), device=device)
    idx = torch.arange(patch_len, device=device).unsqueeze(0)  # [1, P]
    mask = idx < r.unsqueeze(1)                                # [B, P] True where masked
    return mask

def apply_mask_to_first_patch(x_patched: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x_patched: [B, num_patches, patch_len] float tensor
    mask     : [B, patch_len] bool tensor from sample_first_patch_mask
    Zeroes masked positions in the FIRST patch. In a full trainer you would also
    carry this as an attention/padding mask so those tokens are ignored.
    """
    if x_patched.ndim != 3:
        raise ValueError("x_patched must be [B, num_patches, patch_len].")
    B, num_patches, P = x_patched.shape
    if mask.shape != (B, P):
        raise ValueError("mask must be [B, patch_len].")
    x = x_patched.clone()
    # broadcast mask to [B, 1, P] and zero masked values in the first patch only
    x[:, 0, :] = x[:, 0, :].masked_fill(mask, 0.0)
    return x