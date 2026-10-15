# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks specific to the QBot mapless-navigation task.

The contracts every task shares -- declared spaces, finite signals, done
shapes -- are in test_env_smoke.py, which discovers tasks automatically. This
module holds only what is true of this one env and nothing else.

Run from the repository root, inside the Isaac Lab Python environment:

    python -m pytest tests/test_qbot_mapless_nav.py -v
"""

import gymnasium as gym
import pytest
import torch

import omni.usd

from isaaclab_tasks.utils import parse_env_cfg

import quanser_rl_lab.tasks  # noqa: F401

NUM_ENVS = 2
NUM_STEPS = 100
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TASKS = ["Quanser-QBot-Mapless-Nav-v0", "Quanser-QBot-Mapless-Nav-Play-v0"]


@pytest.mark.parametrize("task_name", TASKS)
def test_qbot_observation_contract(task_name):
    omni.usd.get_context().new_stage()
    env_cfg = parse_env_cfg(task_name, device=DEVICE, num_envs=NUM_ENVS)
    env = gym.make(task_name, cfg=env_cfg)
    try:
        cfg = env.unwrapped.cfg
        # The task's own layout rule: one channel per lidar ray, then goal
        # distance, goal heading, two wheel speeds and the gyro.
        obs_dim = cfg.num_lidar_rays + 5

        obs, _ = env.reset()
        assert obs["policy"].shape == (NUM_ENVS, obs_dim)

        # -Play exists to evaluate at full density, so the curriculum must be
        # off from the first episode rather than ramping from 2 cones.
        if task_name.endswith("-Play-v0"):
            assert env.unwrapped._active_num_obstacles == cfg.num_obstacles
        else:
            assert env.unwrapped._active_num_obstacles == cfg.curriculum_start_obstacles

        for _ in range(NUM_STEPS):
            actions = 2.0 * torch.rand(NUM_ENVS, 2, device=env.unwrapped.device) - 1.0
            obs, reward, terminated, truncated, _ = env.step(actions)
            policy_obs = obs["policy"]

            assert policy_obs.shape == (NUM_ENVS, obs_dim)
            # This env normalises every channel, which is its own choice and
            # not a framework guarantee -- a manager-based task would fail
            # this. Anything outside [-1, 1] means a normalisation constant no
            # longer covers its channel's range; the goal-distance channel is
            # the one that breaks first, since it is divided by lidar_max_range
            # and is not clamped.
            assert policy_obs.abs().max() <= 1.0 + 1e-6
    finally:
        env.close()
