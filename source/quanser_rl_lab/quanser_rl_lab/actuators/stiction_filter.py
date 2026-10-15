# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stiction filter for wheel commands.

Models the static-friction delay: a wheel locks when the command is zero AND
it has coasted to a stop, and a locked wheel stays blocked until non-zero
commands have been held for delay_steps consecutive steps.

Stateful per env. apply() runs once per physics substep; reset() must be
called for every env that resets.
"""

import torch


class StictionFilter:
    def __init__(
        self,
        num_envs: int,
        num_wheels: int = 2,
        delay_steps: int = 30,
        cmd_threshold: float = 0.001,
        vel_threshold: float = 0.1,
        device: str = "cpu",
    ):
        """delay_steps is counted in physics substeps, not control steps."""
        self.num_envs = num_envs
        self.num_wheels = num_wheels
        self.delay_steps = delay_steps
        self.cmd_threshold = cmd_threshold
        self.vel_threshold = vel_threshold
        self.device = device

        # Consecutive non-zero-command substeps per wheel, while locked.
        self.nonzero_counter = torch.zeros(num_envs, num_wheels, device=device, dtype=torch.long)

        # Whether each wheel has overcome stiction.
        self.unlocked = torch.zeros(num_envs, num_wheels, device=device, dtype=torch.bool)

    def apply(self, action: torch.Tensor, wheel_vel: torch.Tensor) -> torch.Tensor:
        """Gate raw wheel commands (num_envs, num_wheels) through the model."""
        cmd_nonzero = action.abs() > self.cmd_threshold
        wheel_stopped = wheel_vel.abs() < self.vel_threshold

        # A wheel coasting under a zero command stays unlocked.
        should_lock = ~cmd_nonzero & wheel_stopped
        self.unlocked = self.unlocked & ~should_lock
        self.nonzero_counter = torch.where(should_lock, torch.zeros_like(self.nonzero_counter), self.nonzero_counter)

        counting = cmd_nonzero & ~self.unlocked
        self.nonzero_counter = torch.where(counting, self.nonzero_counter + 1, self.nonzero_counter)

        just_unlocked = self.nonzero_counter >= self.delay_steps
        self.unlocked = self.unlocked | just_unlocked

        filtered = torch.where(self.unlocked, action, torch.zeros_like(action))
        return filtered

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset state for specific environments, or all if env_ids is None."""
        if env_ids is None:
            self.nonzero_counter.zero_()
            self.unlocked.zero_()
        else:
            self.nonzero_counter[env_ids] = 0
            self.unlocked[env_ids] = False
