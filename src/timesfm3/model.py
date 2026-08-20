"""TimesFM3 PyTorch model (inference only).

Supports:
  - forward(): equivalent to Flax __call__, full-sequence forward pass.
  - decode(): non-autoregressive and cached decoding with frozen stats and
    configurable output_patch_len.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from . import configs
from . import cpm_revin_refine as cpm_revin_refine_lib
from . import dense
from . import transformations
from . import transformer
from . import util


class TimesFM3Torch(nn.Module):
  """PyTorch inference-only implementation of TimesFM3.

  Attributes:
    input_patch_len: The length of each input patch.
    output_patch_len: The length of the output forecast for each patch.
    quantiles: A list of quantiles to predict.
    residual_block_config: Configuration for the pre-transformer residual block.
    transformer_config: Configuration for the stacked transformers.
    use_variate_attention: Whether to use variate attention.
    value_clip: Absolute value to clip input values to.
    use_stitching: Whether to use stitching for predictions.
    use_linear_detrending: Whether to apply linear detrending on context.
    linear_detrending_threshold: Ratio threshold for applying linear detrending.
    use_iterative_cpm_revin: Whether to use iterative RevIN refinement.
    use_frozen_running_stats: Whether running stats freeze at context boundary.
  """

  def __init__(
      self,
      input_patch_len: int = 32,
      output_patch_len: int = 64,
      quantiles: list[float] | None = None,
      residual_block_config: configs.ResidualBlockConfig | None = None,
      transformer_config: configs.StackedTransformersConfig | None = None,
      use_variate_attention: bool = True,
      value_clip: float = 1e20,
      use_stitching: bool = True,
      use_linear_detrending: bool = True,
      linear_detrending_threshold: float = 0.5,
      use_iterative_cpm_revin: bool = True,
      use_frozen_running_stats: bool = False,
      input_transform: str = "identity",
  ):
    super().__init__()
    if quantiles is None:
      quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if residual_block_config is None:
      residual_block_config = configs.ResidualBlockConfig(
          hidden_dims=1280,
          output_dims=1280,
          use_bias=False,
          activation="relu",
      )
    if transformer_config is None:
      transformer_config = configs.StackedTransformersConfig(
          num_layers=20,
          transformer=configs.TransformerConfig(
              model_dims=1280,
              hidden_dims=1280,
              num_heads=16,
              attention_norm="rms",
              feedforward_norm="rms",
              qk_norm="rms",
              use_rope_seq=True,
              use_rope_var=False,
              use_bias=False,
              ff_activation="relu",
              deterministic=True,
          ),
      )
    if output_patch_len % input_patch_len != 0:
      raise ValueError(
          f"Output patch len {output_patch_len} must be a multiple of"
          f" input patch len {input_patch_len}."
      )
    if (
        residual_block_config.output_dims
        != transformer_config.transformer.model_dims
    ):
      raise ValueError(
          "ResidualBlock output_dims must match Transformer model_dims."
      )

    self.input_patch_len = input_patch_len
    self.output_patch_len = output_patch_len
    self.quantiles = quantiles
    self.num_quantiles = len(quantiles)
    self.rolls = output_patch_len // input_patch_len
    self.residual_block_config = residual_block_config
    self.transformer_config = transformer_config
    self.use_variate_attention = use_variate_attention
    self.value_clip = value_clip
    self.use_stitching = use_stitching
    self.use_linear_detrending = use_linear_detrending
    self.linear_detrending_threshold = linear_detrending_threshold
    self.use_iterative_cpm_revin = use_iterative_cpm_revin
    self.use_frozen_running_stats = use_frozen_running_stats
    self.input_transform = input_transform

    if self.use_stitching:
      if self.output_patch_len <= self.input_patch_len:
        raise ValueError(
            "use_stitching requires output_patch_len > input_patch_len"
        )
      self._stitching_extract_len = min(
          2 * self.input_patch_len, self.output_patch_len
      )

    self.pre_transformer_resblock = dense.ResidualBlock(
        config=residual_block_config
    )
    self.transformer_stack = transformer.StackedMixingTransformer(
        config=transformer_config,
        use_variate_attention=use_variate_attention,
    )

    self.output_head = nn.Linear(
        transformer_config.transformer.model_dims,
        output_patch_len * self.num_quantiles,
        bias=True,
    )

  def _preprocess(
      self,
      values: torch.Tensor,
      masks: torch.Tensor,
      patch_is_target: torch.Tensor,
      freeze_after: int | None = None,
      patch_cpm_mask: torch.Tensor | None = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      tuple[torch.Tensor, torch.Tensor],
      torch.Tensor,
  ]:
    """Applies preprocessing: RevIN, masking, future covariates, ResBlock.

    Args:
      values: (b, v, n, p).
      masks: (b, v, n, p) bool. True=masked.
      patch_is_target: (b, v, n) bool.
      freeze_after: Optional patch index after which running stats freeze.
      patch_cpm_mask: (b, n) bool or None. If given, target variates at True
        positions are additionally masked (used for horizon CPM masking).

    Returns:
      (resblock_input, resblock_output, patch_mask, (running_mean, running_std),
      running_n)
    """
    running_n, running_mean, running_std = util.get_running_stats(
        values, masks
    )
    if freeze_after is not None:
      _, _, n, _ = values.shape
      if 0 <= freeze_after < n - 1:
        running_mean[:, :, freeze_after + 1 :] = running_mean[
            :, :, freeze_after : freeze_after + 1
        ]
        running_std[:, :, freeze_after + 1 :] = running_std[
            :, :, freeze_after : freeze_after + 1
        ]

    # Apply CPM mask: mask target variates at CPM positions.
    if patch_cpm_mask is not None:
      cpm_bvnp = patch_cpm_mask[:, None, :, None]  # (b, 1, n, 1)
      cpm_target_only = cpm_bvnp & patch_is_target.unsqueeze(-1)
      masks = masks | cpm_target_only

    values_bvnp = util.revin(values, running_mean, running_std, reverse=False)
    values_bvnp = torch.where(masks, 0.0, values_bvnp)

    # Roll values to get future covariate patches
    values_fcov, wrap_mask = util.get_output_patch_via_roll(values, self.rolls)
    values_fcov = util.revin(
        values_fcov, running_mean, running_std, reverse=False
    )

    # Roll the (CPM-modified) masks for future covariate masking.
    masks_fcov_raw, _ = util.get_output_patch_via_roll(masks, self.rolls)
    masks_fcov = masks_fcov_raw | patch_is_target.unsqueeze(-1) | wrap_mask
    values_fcov = torch.where(masks_fcov, 0.0, values_fcov)

    values_cat = torch.cat([values_bvnp, values_fcov], dim=-1)
    masks_cat = torch.cat([masks, masks_fcov], dim=-1)

    resblock_input = torch.cat([values_cat, masks_cat.float()], dim=-1)
    resblock_output = self.pre_transformer_resblock(resblock_input)

    # Patch mask: a patch is fully masked if ALL points are masked
    patch_mask_bvn = masks_cat.all(dim=3)

    return (
        resblock_input,
        resblock_output,
        patch_mask_bvn,
        (running_mean, running_std),
        running_n,
    )

  def forward(
      self,
      inputs: dict[str, Any],
      freeze_after: int | None = None,
      patch_cpm_mask: torch.Tensor | None = None,
      return_aux_outputs: bool = False,
  ) -> dict[str, Any]:
    """Full-sequence forward pass (equivalent to Flax __call__).

    Args:
      inputs: Dictionary with keys: - "values": (b, v, n, p) float - "masks":
        (b, v, n, p) bool - "patch_is_target": (b, v, n) bool
      freeze_after: Optional patch index after which running stats freeze.
      patch_cpm_mask: (b, n) bool or None. Horizon CPM mask.
      return_aux_outputs: Whether to return auxiliary outputs.

    Returns:
      Dictionary with "logits" of shape (b, v, n, output_patch_len,
      num_quantiles).
    """
    values = inputs["values"]
    values = torch.nan_to_num(values, nan=0.0)
    values = torch.clamp(values, -self.value_clip, self.value_clip)
    masks = inputs["masks"].bool()
    patch_is_target = inputs["patch_is_target"]

    _, _, _, p = values.shape
    if p != self.input_patch_len:
      raise ValueError(
          f"Input patch_len {p} != model input_patch_len {self.input_patch_len}"
      )

    # Preprocessing & ResBlock
    (
        resblock_input,
        transformer_input,
        transformer_patch_mask,
        revin_stats,
        running_n,
    ) = self._preprocess(
        values,
        masks,
        patch_is_target,
        freeze_after=freeze_after,
        patch_cpm_mask=patch_cpm_mask,
    )

    # Transformer
    # At inference, only mask *leading* fully-masked patches (left-padding).
    # Flax __call__ does: effective_patch_mask = cumprod(mask, axis=2) when
    # not training.  This keeps horizon patches (which are fully masked but
    # come after valid context) visible to attention.
    effective_patch_mask = torch.cumprod(
        transformer_patch_mask.int(), dim=2
    ).bool()
    transformer_output, _, seq_attn_mask = self.transformer_stack(
        transformer_input,
        effective_patch_mask,
    )

    # Output head
    raw_logits = self.output_head(transformer_output)
    revin_mean, revin_std = revin_stats

    if self.use_iterative_cpm_revin and patch_cpm_mask is not None:
      refined_mu, refined_sigma = (
          cpm_revin_refine_lib.cpm_iterative_revin_refine(
              raw_logits,
              revin_n=running_n,
              revin_mu=revin_mean,
              revin_sigma=revin_std,
              patch_cpm_mask=patch_cpm_mask,
              median_q_idx=self.num_quantiles // 2,
              rolls=self.rolls,
              patch_len=self.input_patch_len,
              num_quantiles=self.num_quantiles,
              value_clip=self.value_clip,
          )
      )
      cpm_bvn = patch_cpm_mask.unsqueeze(1)  # (b, 1, n)
      revin_mean = torch.where(cpm_bvn, refined_mu, revin_mean)
      revin_std = torch.where(cpm_bvn, refined_sigma, revin_std)

    revin_logits = util.revin(raw_logits, revin_mean, revin_std, reverse=True)
    clipped_logits = torch.clamp(
        revin_logits, -self.value_clip, self.value_clip
    )

    # Reshape: (b, v, n, o*q) -> (b, v, n, o, q)
    b, v, n_patches = clipped_logits.shape[:3]
    final_logits = clipped_logits.view(
        b, v, n_patches, self.output_patch_len, self.num_quantiles
    )

    outputs = {"logits": final_logits, "revin_stats": revin_stats}

    if return_aux_outputs:
      outputs["__call__:resblock_input"] = resblock_input
      outputs["__call__:transformer_input"] = transformer_input
      outputs["__call__:seq_attn_mask"] = seq_attn_mask
      outputs["__call__:transformer_output"] = transformer_output

    return outputs

  @torch.no_grad()
  def decode(
      self,
      target: torch.Tensor,
      horizon: int = 0,
      past_only_covariates: torch.Tensor | None = None,
      past_future_covariates: torch.Tensor | None = None,
      target_mask: torch.Tensor | None = None,
      past_only_mask: torch.Tensor | None = None,
      past_future_mask: torch.Tensor | None = None,
      mask: torch.Tensor | None = None,
      return_aux_outputs: bool = False,
  ) -> Any:
    """Non-autoregressive single-pass decoding for TimesFM3.

    Args:
      target: (b, u, context_len).
      horizon: Forecast horizon. Inferred if past_future_covariates given.
      past_only_covariates: (b, v_po, context_len) or None.
      past_future_covariates: (b, w, context_len+horizon) or None.
      target_mask: (b, u, context_len) bool or None.
      past_only_mask: (b, v_po, context_len) bool or None.
      past_future_mask: (b, w, context_len+horizon) bool or None.
      mask: (b, context_len) global bool mask or None.
      return_aux_outputs: If True, return (logits, aux_dict).

    Returns:
      Logits of shape (b, num_variates, horizon, num_quantiles).
    """
    device = target.device
    batch_size, num_target, context = target.shape

    if past_future_covariates is not None:
      horizon = past_future_covariates.shape[-1] - context
    if horizon <= 0:
      raise ValueError("Decode function requires horizon > 0.")

    # 1. Pad context to multiple of input_patch_len
    ctx_padding = (
        self.input_patch_len - (context % self.input_patch_len)
    ) % self.input_patch_len
    if ctx_padding > 0:
      target = torch.nn.functional.pad(target, (ctx_padding, 0))
      if mask is not None:
        mask = torch.nn.functional.pad(mask, (ctx_padding, 0), value=True)
      if past_only_covariates is not None:
        past_only_covariates = torch.nn.functional.pad(
            past_only_covariates, (ctx_padding, 0)
        )
      if past_future_covariates is not None:
        past_future_covariates = torch.nn.functional.pad(
            past_future_covariates, (ctx_padding, 0)
        )
      if target_mask is not None:
        target_mask = torch.nn.functional.pad(
            target_mask, (ctx_padding, 0), value=True
        )
      if past_only_mask is not None:
        past_only_mask = torch.nn.functional.pad(
            past_only_mask, (ctx_padding, 0), value=True
        )
      if past_future_mask is not None:
        past_future_mask = torch.nn.functional.pad(
            past_future_mask, (ctx_padding, 0), value=True
        )
      context = context + ctx_padding

    if mask is None:
      mask = torch.zeros(batch_size, context, dtype=torch.bool, device=device)
      if ctx_padding > 0:
        mask[:, :ctx_padding] = True

    # 2. Pad horizon
    if self.use_stitching:
      extract_len = self._stitching_extract_len
      overlap = extract_len - self.input_patch_len
      num_forecast_patches = max(
          math.ceil((horizon - overlap) / self.input_patch_len), 1
      )
      num_horizon_patches = num_forecast_patches + self.rolls - 1
      padded_horizon = num_horizon_patches * self.input_patch_len
      hor_padding = padded_horizon - horizon
    else:
      hor_padding = (-horizon) % self.output_patch_len
      padded_horizon = horizon + hor_padding
      num_horizon_patches = padded_horizon // self.input_patch_len
    num_context_patches = context // self.input_patch_len

    # 3. Build context & horizon inputs
    if target_mask is None:
      target_mask = torch.zeros_like(target, dtype=torch.bool)
    target_mask = target_mask | mask.unsqueeze(1)

    all_ctx_vals = [target]
    all_ctx_masks = [target_mask]
    num_past_only = 0
    if past_only_covariates is not None:
      num_past_only = past_only_covariates.shape[1]
      if past_only_mask is None:
        past_only_mask = torch.zeros_like(
            past_only_covariates, dtype=torch.bool
        )
      all_ctx_vals.append(past_only_covariates)
      all_ctx_masks.append(past_only_mask | mask.unsqueeze(1))
    if past_future_covariates is not None:
      if past_future_mask is None:
        past_future_mask = torch.zeros_like(
            past_future_covariates, dtype=torch.bool
        )
      all_ctx_vals.append(past_future_covariates[..., :context])
      all_ctx_masks.append(past_future_mask[..., :context] | mask.unsqueeze(1))

    ctx_vals = torch.cat(all_ctx_vals, dim=1)
    ctx_masks = torch.cat(all_ctx_masks, dim=1)

    if self.use_linear_detrending:
      t_ctx = torch.arange(
          -(context - 1), 1, dtype=torch.float32, device=device
      )
      t_ctx_bvc = t_ctx[None, None, :]
      t_ctx_bvc_normalized = t_ctx_bvc / context

      valid = ~ctx_masks
      n_v = valid.float().sum(dim=-1, keepdim=True)
      sum_t = torch.where(valid, t_ctx_bvc_normalized, 0.0).sum(
          dim=-1, keepdim=True
      )
      sum_t2 = torch.where(valid, t_ctx_bvc_normalized**2, 0.0).sum(
          dim=-1, keepdim=True
      )
      sum_y = torch.where(valid, ctx_vals, 0.0).sum(dim=-1, keepdim=True)
      sum_ty = torch.where(valid, t_ctx_bvc_normalized * ctx_vals, 0.0).sum(
          dim=-1, keepdim=True
      )

      det = n_v * sum_t2 - sum_t**2
      safe_det = torch.where(det == 0.0, 1.0, det)
      m_trend = torch.where(
          det == 0.0, 0.0, (n_v * sum_ty - sum_t * sum_y) / safe_det
      )
      c_trend = torch.where(
          det == 0.0,
          torch.where(n_v > 0, sum_y / torch.clamp_min(n_v, 1.0), 0.0),
          (sum_y - m_trend * sum_t) / torch.clamp_min(n_v, 1.0),
      )

      ctx_vals_detrended = ctx_vals - (
          m_trend * t_ctx_bvc_normalized + c_trend
      )

      mean_y = sum_y / torch.clamp_min(n_v, 1.0)
      sum_y2 = torch.where(valid, ctx_vals**2, 0.0).sum(dim=-1, keepdim=True)
      var_orig = torch.clamp_min(
          sum_y2 / torch.clamp_min(n_v, 1.0) - mean_y**2, 0.0
      )
      std_orig = torch.sqrt(var_orig)

      sum_yd = torch.where(valid, ctx_vals_detrended, 0.0).sum(
          dim=-1, keepdim=True
      )
      mean_yd = sum_yd / torch.clamp_min(n_v, 1.0)
      sum_yd2 = torch.where(valid, ctx_vals_detrended**2, 0.0).sum(
          dim=-1, keepdim=True
      )
      var_det = torch.clamp_min(
          sum_yd2 / torch.clamp_min(n_v, 1.0) - mean_yd**2, 0.0
      )
      std_det = torch.sqrt(var_det)

      apply_detrend = std_det < self.linear_detrending_threshold * std_orig
      ctx_vals = torch.where(apply_detrend, ctx_vals_detrended, ctx_vals)
    else:
      num_variates = ctx_vals.shape[1]
      m_trend = torch.zeros(
          (batch_size, num_variates, 1), dtype=torch.float32, device=device
      )
      c_trend = torch.zeros(
          (batch_size, num_variates, 1), dtype=torch.float32, device=device
      )
      apply_detrend = torch.zeros(
          (batch_size, num_variates, 1), dtype=torch.bool, device=device
      )

    ctx_vals = torch.where(ctx_masks, 0.0, ctx_vals)

    all_hor_vals = [
        torch.zeros(batch_size, num_target, padded_horizon, device=device),
        torch.zeros(batch_size, num_past_only, padded_horizon, device=device),
    ]
    all_hor_masks = [
        torch.ones(
            batch_size,
            num_target,
            padded_horizon,
            dtype=torch.bool,
            device=device,
        ),
        torch.ones(
            batch_size,
            num_past_only,
            padded_horizon,
            dtype=torch.bool,
            device=device,
        ),
    ]

    if past_future_covariates is not None:
      if past_future_mask is None:
        past_future_mask = torch.zeros_like(
            past_future_covariates, dtype=torch.bool
        )
      pf_future_vals = past_future_covariates[..., context : context + horizon]
      pf_future_masks = past_future_mask[..., context : context + horizon]
      if self.use_linear_detrending:
        m_pf = m_trend[:, num_target + num_past_only :, :]
        c_pf = c_trend[:, num_target + num_past_only :, :]
        apply_detrend_pf = apply_detrend[:, num_target + num_past_only :, :]
        t_hor_pf = torch.arange(
            1, horizon + 1, dtype=torch.float32, device=device
        )[None, None, :]
        t_hor_pf_normalized = t_hor_pf / context
        pf_trend_hor = m_pf * t_hor_pf_normalized + c_pf
        pf_future_vals = torch.where(
            apply_detrend_pf, pf_future_vals - pf_trend_hor, pf_future_vals
        )
      pf_future_vals = torch.where(pf_future_masks, 0.0, pf_future_vals)
      if hor_padding > 0:
        pf_future_vals = torch.nn.functional.pad(
            pf_future_vals, (0, hor_padding)
        )
        pf_future_masks = torch.nn.functional.pad(
            pf_future_masks, (0, hor_padding), value=True
        )
      all_hor_vals.append(pf_future_vals)
      all_hor_masks.append(pf_future_masks)

    hor_vals = torch.cat(all_hor_vals, dim=1)
    hor_masks = torch.cat(all_hor_masks, dim=1)

    all_vals = torch.cat([ctx_vals, hor_vals], dim=-1)
    all_masks = torch.cat([ctx_masks, hor_masks], dim=-1)

    num_variates = all_vals.shape[1]
    patch_is_target = torch.zeros(
        (batch_size, num_variates, num_context_patches + num_horizon_patches),
        dtype=torch.bool,
        device=device,
    )
    patch_is_target[:, : num_target + num_past_only, :] = True

    # Reshape values & masks to patched shape (b, v, n, p)
    values_bvnp = all_vals.reshape(
        batch_size, num_variates, -1, self.input_patch_len
    )
    masks_bvnp = all_masks.reshape(
        batch_size, num_variates, -1, self.input_patch_len
    )

    inputs = {
        "values": values_bvnp,
        "masks": masks_bvnp,
        "patch_is_target": patch_is_target,
    }

    # Build horizon CPM mask: context=False, horizon=True.
    num_total_patches = num_context_patches + num_horizon_patches
    horizon_cpm_mask = torch.zeros(
        batch_size, num_total_patches, dtype=torch.bool, device=device
    )
    horizon_cpm_mask[:, num_context_patches:] = True

    freeze_after = (
        num_context_patches - 1 if self.use_frozen_running_stats else None
    )
    forward_out = self.forward(
        inputs,
        freeze_after=freeze_after,
        patch_cpm_mask=horizon_cpm_mask,
        return_aux_outputs=return_aux_outputs,
    )
    logits = forward_out["logits"]  # (b, v, n, output_patch_len, num_quantiles)

    if self.use_stitching:
      extract_len = self._stitching_extract_len
      overlap = extract_len - self.input_patch_len
      num_forecast_patches = max(
          math.ceil((horizon - overlap) / self.input_patch_len), 1
      )
      forecast_indices = torch.arange(
          num_forecast_patches, device=device
      ) + (num_context_patches - 1)
      patch_preds = logits[:, :, forecast_indices, :extract_len, :]
      horizon_logits = util.stitch_patches(
          patch_preds,
          self.input_patch_len,
      )[:, :, :horizon, :]
    else:
      num_forecast_chunks = padded_horizon // self.output_patch_len
      forecast_indices = torch.arange(
          num_forecast_chunks, device=device
      ) * self.rolls + (num_context_patches - 1)
      forecast_logits = logits[:, :, forecast_indices, :, :]
      horizon_logits = forecast_logits.reshape(
          batch_size, num_variates, -1, self.num_quantiles
      )[:, :, :horizon, :]

    if self.use_linear_detrending:
      t_forecast = torch.arange(
          1, horizon + 1, dtype=torch.float32, device=device
      )
      t_forecast_normalized = t_forecast / context
      trend_forecast = (
          m_trend[:, :, 0, None] * t_forecast_normalized[None, None, :]
          + c_trend[:, :, 0, None]
      )
      trend_forecast = torch.where(
          apply_detrend[:, :, 0, None], trend_forecast, 0.0
      )
      horizon_logits = horizon_logits + trend_forecast[:, :, :, None]

    if return_aux_outputs:
      return horizon_logits, forward_out
    return horizon_logits
