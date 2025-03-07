"""Implements different finetuning strategies for the TimesFM model.

This module provides abstract and concrete implementations of finetuning strategies
that can be applied to TimesFM models. It includes full finetuning and linear probing
approaches.

The module contains the following classes:
    - FinetuningStrategy: Abstract base class defining the strategy interface
    - FullFinetuningStrategy: Strategy that allows training of all model parameters
    - LinearProbingStrategy: Strategy that freezes base model and adds trainable linear head
    - TimesFMLinearProbingWrapper: Wrapper class implementing linear probing functionality

Typical usage example:
    strategy = LinearProbingStrategy()
    model = strategy.configure_model(base_model)
    # model is now configured for linear probing
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FinetuningStrategy(ABC):

  @abstractmethod
  def configure_model(self, model: nn.Module) -> nn.Module:
    """Apply any freezing / structural modifications and return updated model"""
    pass


class FullFinetuningStrategy(FinetuningStrategy):
  """Leaves the entire model trainable."""

  def configure_model(self, model: nn.Module) -> nn.Module:
    return model


class LinearProbingStrategy(FinetuningStrategy):
  """
    Freezes the original model, adds a new linear head,
    and returns a wrapper that only trains the new head.
  """

  def configure_model(self, model: nn.Module) -> nn.Module:

    # todo: not very good, because we hardcode the arch from patched_decoder.py. Make it more flexible in next PR
    output_dim = model.stacked_transformer.layers[
        -1].mlp.down_proj.out_features

    wrapper = TimesFMLinearProbingWrapper(base_model=model,
                                          output_dim=output_dim)
    return wrapper


class TimesFMLinearProbingWrapper(nn.Module):
  """
    Wraps a TimesFM model for linear probing:
      1) Freezes the base model parameters
      2) Adds a trainable linear head as the only learnable component
  """

  def __init__(self, base_model: nn.Module, output_dim: int):
    super().__init__()
    self.base_model = base_model
    self.output_dim = output_dim

    for param in self.base_model.parameters():
      param.requires_grad = False

    self.linear_head = nn.Sequential(nn.Linear(output_dim,
                                               output_dim * 2), nn.ReLU(),
                                     nn.Linear(output_dim * 2, output_dim))

  def forward(self, x_context, x_padding, freq):
    with torch.no_grad():
      predictions = self.base_model(x_context, x_padding, freq)
    
    predictions_mean = predictions[..., 0]
    last_patch_pred = predictions_mean[:, -1, :]
    print(last_patch_pred.shape)
    transformed = self.linear_head(last_patch_pred)
    print(transformed.shape)
    output = last_patch_pred.clone()
    output[:, -1:, :] = transformed
    print(output.shape)
    return output
