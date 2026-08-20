"""Standalone iterative RevIN refinement for CPM-masked patches in PyTorch.

Extracted so it can be tested independently of the TimesFM3 model.
"""

from __future__ import annotations

import torch

from . import util


def cpm_iterative_revin_refine(
    raw_logits: torch.Tensor,
    revin_n: torch.Tensor,
    revin_mu: torch.Tensor,
    revin_sigma: torch.Tensor,
    patch_cpm_mask: torch.Tensor,
    median_q_idx: int,
    rolls: int,
    patch_len: int,
    num_quantiles: int,
    value_clip: float = 1e9,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Refines RevIN stats at CPM-masked patches via iterative estimation.

  For each CPM-masked position p the currently frozen stats (from the last
  observed patch before the CPM region) are replaced with stats that also
  incorporate model-estimated values for all CPM patches that precede p.

  Args:
    raw_logits: Output of the output head, shape (b, v, n, output_patch_len *
      num_quantiles). In RevIN-normalised space, before reverse RevIN.
    revin_n: Count of valid (unmasked) values accumulated at each patch
      position, shape (b, v, n).
    revin_mu: Running mean per position, (b, v, n).
    revin_sigma: Running std per position, (b, v, n).
    patch_cpm_mask: Boolean mask, True = CPM-masked patch, shape (b, n).
    median_q_idx: Index into quantiles selecting the median quantile used as
      point estimate (typically num_quantiles // 2).
    rolls: Number of output patches per input patch (output_patch_len //
      patch_len).
    patch_len: Length of each input patch.
    num_quantiles: Total number of quantile heads.
    value_clip: Absolute bound for clamping estimated values after reverse
      RevIN.

  Returns:
    Tuple (refined_mu, refined_sigma), each shape (b, v, n).
    Non-CPM positions are identical to revin_mu / revin_sigma.
    CPM positions incorporate estimates of all preceding CPM patches in
    the same block (and all estimates from earlier blocks in the segment).
  """
  b, v, n_patches, _ = raw_logits.shape
  device = raw_logits.device

  # Reshape and slice raw_logits to keep only the median quantile.
  # (b, v, n, oq) -> (b, v, n, rolls, patch_len, num_quantiles)
  # -> (b, v, n, rolls, patch_len)
  median_logits = raw_logits.reshape(
      b, v, n_patches, rolls, patch_len, num_quantiles
  )[:, :, :, :, :, median_q_idx]

  # Initialise carry with zeros.
  carry_n = torch.zeros((b, v), dtype=torch.float32, device=device)
  carry_mu = torch.zeros((b, v), dtype=torch.float32, device=device)
  carry_sigma = torch.zeros((b, v), dtype=torch.float32, device=device)
  anchor_predicted_values = torch.zeros(
      (b, v, rolls, patch_len), dtype=torch.float32, device=device
  )
  block_offset = torch.zeros((b,), dtype=torch.long, device=device)

  refined_mu_list = []
  refined_sigma_list = []

  step_masks = torch.zeros((b, v, patch_len), dtype=torch.bool, device=device)

  for i in range(n_patches):
    actual_n = revin_n[:, :, i]
    actual_mu = revin_mu[:, :, i]
    actual_sigma = revin_sigma[:, :, i]
    current_step_logits = median_logits[:, :, i]
    is_cpm = patch_cpm_mask[:, i : i + 1]  # (b, 1)

    # Select the block_offset[b]-th patch for each batch element
    offset_onehot = torch.eq(
        torch.arange(rolls, device=device).unsqueeze(0),
        block_offset.unsqueeze(1),
    ).float()
    predicted_values_step = torch.einsum(
        "br,bvrp->bvp", offset_onehot, anchor_predicted_values
    )

    # Update running stats with the estimated patch.
    new_n, new_mu, new_sigma = util.update_running_stats(
        carry_n, carry_mu, carry_sigma, predicted_values_step, step_masks
    )

    out_n = torch.where(is_cpm, new_n, actual_n)
    out_mu = torch.where(is_cpm, new_mu, actual_mu)
    out_sigma = torch.where(is_cpm, new_sigma, actual_sigma)

    # Advance block_offset: +1 (mod rolls) for CPM, reset to 0 for non-CPM.
    new_block_offset = torch.where(
        is_cpm.squeeze(-1),
        (block_offset + 1) % rolls,
        torch.zeros_like(block_offset),
    )

    should_update_anchor = torch.eq(new_block_offset, 0)

    # Pre-calculate predicted values for the new anchor.
    step_predicted_values = util.revin(
        current_step_logits, out_mu, out_sigma, reverse=True
    )
    step_predicted_values = torch.clamp(
        step_predicted_values, -value_clip, value_clip
    )

    new_anchor_predicted_values = torch.where(
        should_update_anchor.view(b, 1, 1, 1),
        step_predicted_values,
        anchor_predicted_values,
    )

    carry_n = out_n
    carry_mu = out_mu
    carry_sigma = out_sigma
    anchor_predicted_values = new_anchor_predicted_values
    block_offset = new_block_offset

    refined_mu_list.append(out_mu)
    refined_sigma_list.append(out_sigma)

  refined_mu = torch.stack(refined_mu_list, dim=2)
  refined_sigma = torch.stack(refined_sigma_list, dim=2)
  return refined_mu, refined_sigma
