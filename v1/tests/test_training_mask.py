# tests/test_training_mask.py
import torch
from timesfm.train_utils.masking import sample_first_patch_mask, apply_mask_to_first_patch

def test_mask_shape_and_bounds():
    B, P = 8, 16
    m = sample_first_patch_mask(P, B, device='cpu')
    assert m.shape == (B, P)
    assert m.dtype == torch.bool
    # Each row should look like [True x r] + [False x (P-r)], i.e., no False->True transitions
    diffs = m[:, 1:].int() - m[:, :-1].int()
    assert not (diffs == 1).any().item()

def test_apply_mask():
    B, num_patches, P = 4, 3, 8
    x = torch.ones(B, num_patches, P)
    m = torch.zeros(B, P, dtype=torch.bool)
    m[:, :3] = True  # mask first 3 positions in first patch
    y = apply_mask_to_first_patch(x, m)
    assert torch.allclose(y[:, 0, :3], torch.zeros(B, 3))
    assert torch.allclose(y[:, 0, 3:], torch.ones(B, P-3))
    # other patches untouched
    assert torch.allclose(y[:, 1:, :], torch.ones(B, num_patches-1, P))