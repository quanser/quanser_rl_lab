# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gaussian action distribution with a soft cap on std."""

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal


class CappedGaussianDistribution(GaussianDistribution):
    """GaussianDistribution with std soft-capped in update().

    Needed because TanhMeanMLPModel bounds only the *mean*; std is a separate
    parameter that entropy_coef pushes up with nothing pushing back, and it has
    been observed to run away to ~1e18 over a few thousand iterations.

    sample(), log_prob(), entropy, kl_divergence and the mean/std properties are
    all inherited unchanged.
    """

    # 0.3 is set by the clamp-hit rate it implies at the env's action clip, not
    # by what avoids crashing. tanh rather than clamp() so the saturated region
    # still has a usable gradient.
    _MAX_STD = 0.3

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(mlp_output)
        capped_std = self._MAX_STD * torch.tanh(self._distribution.stddev / self._MAX_STD)  # type: ignore
        self._distribution = Normal(self._distribution.mean, capped_std)  # type: ignore
