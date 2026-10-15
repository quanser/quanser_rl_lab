# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Actor model that bounds the Gaussian's mean with a tanh on the MLP's last layer."""

import torch
from rsl_rl.models import MLPModel


class TanhMeanMLPModel(MLPModel):
    """Tanh on the MLP output, which for GaussianDistribution is the mean.

    Bounds the mean to (-1, 1) and nothing else, so log_prob, entropy and
    kl_divergence stay rsl_rl's own Gaussian math. The sample is still
    unbounded: CappedGaussianDistribution limits std, and
    QBotNavEnv._pre_physics_step clamps the scaled action.

    Do not swap in a distribution that squashes the sample instead. Re-scoring
    a stored sample then needs an atanh inversion that is quadratically
    sensitive near +-1, which may crash training.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = torch.nn.Sequential(self.mlp, torch.nn.Tanh())
