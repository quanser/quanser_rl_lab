# Copyright 2025 ROBOTIS CO., LTD.
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
#
# Author: Taehyeong Kim
#
# Modifications copyright 2026 Quanser Consulting Inc., Apache-2.0.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from quanser_rl_lab.assets.qarm_research import QARM_RESEARCH_CFG, verify_usd_available  # isort: skip


##
# Environment configuration
##


@configclass
class QArmResearchReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Fail here, with a message that names the cause, rather than hundreds of
        # lines into Isaac Sim startup as an unrelated-looking prim error.
        verify_usd_available()

        # switch robot to QArm Research (self-collisions enabled, penalized via rewards.arm_contact)
        self.scene.robot = QARM_RESEARCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link6"]

        # Joint and body names are the prim names inside qarm_research.usd.
        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        # override actions
        # scale is a gain, not a bound.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=arm_joint_names,
            scale=0.5,
            use_default_offset=True,
        )
        # override observations policy
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names
        )

        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "link6"


@configclass
class QArmResearchReachEnvCfg_PLAY(QArmResearchReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # Goal sampling is left as the training config has it, so evaluation
        # covers the same two zones the policy was trained on.
