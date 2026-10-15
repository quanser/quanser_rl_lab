# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for every registered task: build it, step it, check the contracts
every Isaac Lab environment owes its caller regardless of robot.

Tasks are discovered from the gym registry, so a newly registered robot is
covered without editing this file. Anything specific to one task belongs in
its own test module, not here.

Run from the repository root, inside the Isaac Lab Python environment:

    python -m pytest tests/test_env_smoke.py -v
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

import omni.usd

from isaaclab_tasks.utils import parse_env_cfg

import quanser_rl_lab.tasks  # noqa: F401 -- registers Quanser tasks

NUM_ENVS = 2
NUM_STEPS = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Same filter list_envs.py uses, so the two never disagree about what exists.
TASKS = sorted(spec.id for spec in gym.registry.values() if "Quanser-" in spec.id)


@pytest.mark.parametrize("task_name", TASKS)
def test_env_obeys_its_declared_spaces(task_name):
    omni.usd.get_context().new_stage()
    env_cfg = parse_env_cfg(task_name, device=DEVICE, num_envs=NUM_ENVS)
    env = gym.make(task_name, cfg=env_cfg)
    try:
        # What the env promises, taken from the env rather than recomputed, so
        # this catches a declared space that disagrees with what is returned.
        obs_dim = int(np.prod(env.unwrapped.single_observation_space["policy"].shape))
        act_shape = (NUM_ENVS, int(np.prod(env.unwrapped.single_action_space.shape)))

        obs, _ = env.reset()
        assert obs["policy"].shape == (NUM_ENVS, obs_dim)

        for _ in range(NUM_STEPS):
            actions = 2.0 * torch.rand(act_shape, device=env.unwrapped.device) - 1.0
            obs, reward, terminated, truncated, _ = env.step(actions)

            assert obs["policy"].shape == (NUM_ENVS, obs_dim)
            assert torch.isfinite(obs["policy"]).all()
            assert reward.shape == (NUM_ENVS,)
            assert torch.isfinite(reward).all()
            assert terminated.shape == truncated.shape == (NUM_ENVS,)
    finally:
        env.close()
