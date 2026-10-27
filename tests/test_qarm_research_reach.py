# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks specific to the QArm Research reach task.

The contracts every task shares -- declared spaces, finite signals, done
shapes -- are in test_env_smoke.py, which discovers tasks automatically. This
module holds only what is true of this one env and nothing else.

Run from the repository root, inside the Isaac Lab Python environment:

    python -m pytest tests/test_qarm_research_reach.py -v
"""

import gymnasium as gym
import pytest
import torch

import omni.usd

from isaaclab_tasks.utils import parse_env_cfg

import quanser_rl_lab.tasks  # noqa: F401

NUM_ENVS = 64
NUM_STEPS = 60
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TASKS = ["Quanser-QArmResearch-Reach-v0", "Quanser-QArmResearch-Reach-Play-v0"]

# 6 joint positions + 6 joint velocities + 7 goal pose + 6 previous actions.
ARM_DOF = 6
POSE_COMMAND_DIM = 7
EXPECTED_OBS_DIM = ARM_DOF + ARM_DOF + POSE_COMMAND_DIM + ARM_DOF


@pytest.mark.parametrize("task_name", TASKS)
def test_qarm_research_observation_layout(task_name):
    """The observation is a concatenation whose parts must stay in step.

    joint_pos and joint_vel are filtered to the six arm joints, so adding a
    gripper action or observing the gripper joints would silently change the
    width. Asserting the decomposition rather than just the total catches that.
    """
    omni.usd.get_context().new_stage()
    env_cfg = parse_env_cfg(task_name, device=DEVICE, num_envs=NUM_ENVS)
    env = gym.make(task_name, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (NUM_ENVS, EXPECTED_OBS_DIM)
        # The arm is commanded; the gripper is actuated but never actioned.
        assert env.unwrapped.single_action_space.shape == (ARM_DOF,)
        assert env.unwrapped.action_manager.total_action_dim == ARM_DOF
    finally:
        env.close()


@pytest.mark.parametrize("task_name", TASKS)
def test_goals_come_from_both_zones(task_name):
    """The two-zone command is what distinguishes this task from upstream reach.

    A plain UniformPoseCommand would pass every other check in this repo, so
    without this the zone split could regress to a single box unnoticed. Zone B
    is the only one with pos_y above 0.12, which makes it identifiable from the
    command alone.
    """
    omni.usd.get_context().new_stage()
    env_cfg = parse_env_cfg(task_name, device=DEVICE, num_envs=NUM_ENVS)
    env = gym.make(task_name, cfg=env_cfg)
    try:
        env.reset()
        seen_a = seen_b = False
        for _ in range(NUM_STEPS):
            actions = torch.zeros((NUM_ENVS, ARM_DOF), device=env.unwrapped.device)
            env.step(actions)
            command = env.unwrapped.command_manager.get_command("ee_pose")
            in_b = command[:, 1] > 0.12
            seen_b |= bool(in_b.any())
            seen_a |= bool((~in_b).any())
        assert seen_a, "no goal was sampled from zone A"
        assert seen_b, "no goal was sampled from zone B"
    finally:
        env.close()


def test_play_variant_disables_observation_noise():
    """-Play-v0 exists so evaluation is not fighting the training noise model."""
    omni.usd.get_context().new_stage()
    train_cfg = parse_env_cfg(TASKS[0], device=DEVICE, num_envs=NUM_ENVS)
    play_cfg = parse_env_cfg(TASKS[1], device=DEVICE, num_envs=NUM_ENVS)
    assert train_cfg.observations.policy.enable_corruption is True
    assert play_cfg.observations.policy.enable_corruption is False
