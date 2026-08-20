"""Dense layers for TimesFM3 PyTorch (inference only)."""

from __future__ import annotations

import torch
from torch import nn

from . import configs
from . import util


class ResidualBlock(nn.Module):
  """Residual block with two linear layers and a linear residual connection.

  Architecture:
    if prenorm == "rms": x_norm = RMSNorm(x)  else: x_norm = x
    hidden = activation(hidden_layer(x_norm))
    output = output_layer(hidden) + residual_layer(x)  [or + x if identity_skip]
  """

  def __init__(self, config: configs.ResidualBlockConfig):
    super().__init__()
    self.config = config

    # Defining placeholder layers in __init__ ensures PyTorch registers them as
    # submodules. This is required for standard parameter tracking, printing/
    # debugging, and correctly propagating device/dtype moves applied to the
    # parent module (e.g. `model.to(device)`) before any forward handles them.
    # They are safely re-initialized/overwritten with correct dimensions during
    # the first forward pass (using `set_input_dims()`) before the matrix
    # multiplication is evaluated, avoiding shape mismatch errors.
    self.hidden_layer = nn.Linear(
        in_features=config.hidden_dims,  # placeholder, set in first forward
        out_features=config.hidden_dims,
        bias=config.use_bias,
    )
    self.output_layer = nn.Linear(
        in_features=config.hidden_dims,
        out_features=config.output_dims,
        bias=config.use_bias,
    )

    if config.identity_skip:
      self.residual_layer = None
    else:
      self.residual_layer = nn.Linear(
          in_features=config.hidden_dims,  # placeholder
          out_features=config.output_dims,
          bias=config.use_bias,
      )

    self.activation = util.get_activation_fn(config.activation)

    if config.prenorm == "rms":
      self.pre_norm = nn.RMSNorm(config.hidden_dims)
    else:
      self.pre_norm = None

    # Mark layers as lazy so input dim gets set on first use
    self._input_dim_set = False

  def set_input_dims(self, input_dim: int) -> None:
    """Reinitialize linear layers with the correct input dimension."""
    if self._input_dim_set:
      return
    device = self.hidden_layer.weight.device
    dtype = self.hidden_layer.weight.dtype

    self.hidden_layer = nn.Linear(
        input_dim, self.config.hidden_dims, bias=self.config.use_bias
    ).to(device=device, dtype=dtype)
    self.output_layer = nn.Linear(
        self.config.hidden_dims,
        self.config.output_dims,
        bias=self.config.use_bias,
    ).to(device=device, dtype=dtype)
    if self.residual_layer is not None:
      self.residual_layer = nn.Linear(
          input_dim, self.config.output_dims, bias=self.config.use_bias
      ).to(device=device, dtype=dtype)
    if self.pre_norm is not None:
      self.pre_norm = nn.RMSNorm(input_dim).to(device=device, dtype=dtype)
    self._input_dim_set = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass. x shape: (b, ..., input_dim)."""
    if not self._input_dim_set:
      self.set_input_dims(x.shape[-1])

    if self.pre_norm is not None:
      hidden_input = self.pre_norm(x)
    else:
      hidden_input = x

    hidden_output = self.activation(self.hidden_layer(hidden_input))

    if self.residual_layer is not None:
      return self.output_layer(hidden_output) + self.residual_layer(x)
    else:
      return self.output_layer(hidden_output) + x
