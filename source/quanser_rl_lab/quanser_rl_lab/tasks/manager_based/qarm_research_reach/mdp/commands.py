# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose command generator that samples from two disjoint zones."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    pass


class TwoZoneUniformPoseCommand(UniformPoseCommand):
    """Uniform pose command sampled from one of two disjoint zones.

    Each time an environment is resampled, a coin flip picks zone A (``cfg.ranges``) or
    zone B (``cfg.ranges_b``). Position and orientation are then drawn from that zone's
    ranges only, so each zone keeps its own narrow orientation band. Everything else
    (buffers, metrics, quaternion handling, debug visualization) is inherited unchanged.
    """

    cfg: TwoZoneUniformPoseCommandCfg

    def __str__(self) -> str:
        msg = "TwoZoneUniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tZone B probability: {self.cfg.zone_b_prob}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        num = len(env_ids)
        use_b = torch.rand(num, device=self.device) < self.cfg.zone_b_prob

        def sample_axis(range_a: tuple[float, float], range_b: tuple[float, float]) -> torch.Tensor:
            a = torch.empty(num, device=self.device).uniform_(*range_a)
            b = torch.empty(num, device=self.device).uniform_(*range_b)
            return torch.where(use_b, b, a)

        ranges_a, ranges_b = self.cfg.ranges, self.cfg.ranges_b
        # -- position
        self.pose_command_b[env_ids, 0] = sample_axis(ranges_a.pos_x, ranges_b.pos_x)
        self.pose_command_b[env_ids, 1] = sample_axis(ranges_a.pos_y, ranges_b.pos_y)
        self.pose_command_b[env_ids, 2] = sample_axis(ranges_a.pos_z, ranges_b.pos_z)
        # -- orientation
        roll = sample_axis(ranges_a.roll, ranges_b.roll)
        pitch = sample_axis(ranges_a.pitch, ranges_b.pitch)
        yaw = sample_axis(ranges_a.yaw, ranges_b.yaw)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class TwoZoneUniformPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for the two-zone uniform pose command generator."""

    class_type: type = TwoZoneUniformPoseCommand

    ranges_b: UniformPoseCommandCfg.Ranges = MISSING
    """Zone B ranges. ``ranges`` (inherited) holds zone A."""

    zone_b_prob: float = 0.5
    """Probability that an environment's goal is sampled from zone B instead of zone A."""
