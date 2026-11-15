# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for an exactly-constant patch.

A patch in which every value is identical has zero variance. ``sqrt`` is finite
at zero but its derivative is not, so the forward pass returns ``0.0`` while the
backward pass returns NaN and the training step is silently discarded.

The constant used here is ``0.5``, chosen deliberately: its sum over a patch is
exactly representable in float32, so the variance is exactly zero on **both**
backends and the test fails before the fix on CPU as well as CUDA. With the
water-quality value ``0.025`` the CPU float32 sum is inexact, leaving a variance
of ~1.9e-09 and a finite gradient, so a CPU-only suite would not see the bug at
all -- that value is covered by the second test, which is CUDA-only for that
reason.
"""

from absl.testing import absltest
import torch

from timesfm.torch import util

_EXACT_CONSTANT = 0.5      # exactly representable: variance is exactly 0 on CPU and CUDA
_CENSORED_VALUE = 0.025    # a sub-detection-limit run reported at the limit (CUDA only)


class ConstantPatchTest(absltest.TestCase):

  def _run(self, device, value):
    torch.manual_seed(0)
    batch, variate, patch = 4, 2, 32
    x = torch.full((batch, variate, patch), value, device=device)
    x.requires_grad_(True)
    n = torch.zeros(batch, variate, device=device)
    mu = torch.zeros(batch, variate, device=device)
    sigma = torch.ones(batch, variate, device=device)
    mask = torch.zeros_like(x, dtype=torch.bool)
    _, (_, _, new_sigma) = util.update_running_stats(n, mu, sigma, x, mask)
    return x, new_sigma

  def test_constant_patch_scale_is_strictly_positive(self):
    """Device independent: a zero variance must not yield a zero scale."""
    _, new_sigma = self._run("cpu", _EXACT_CONSTANT)
    self.assertTrue(torch.isfinite(new_sigma).all())
    self.assertTrue(
        (new_sigma > 0).all(),
        "a constant patch yielded sigma == 0; sqrt(0) has an infinite derivative",
    )

  def test_constant_patch_gradients_are_finite(self):
    x, new_sigma = self._run("cpu", _EXACT_CONSTANT)
    new_sigma.sum().backward()
    self.assertIsNotNone(x.grad)
    self.assertTrue(
        torch.isfinite(x.grad).all(),
        "non-finite gradient from a constant patch; the training step "
        "would be discarded without any diagnostic",
    )

  @absltest.skipUnless(torch.cuda.is_available(), "needs CUDA float32")
  def test_censored_value_gradients_are_finite_on_cuda(self):
    """The water-quality case: sub-detection-limit values reported at the limit."""
    x, new_sigma = self._run("cuda", _CENSORED_VALUE)
    new_sigma.sum().backward()
    self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
  absltest.main()
