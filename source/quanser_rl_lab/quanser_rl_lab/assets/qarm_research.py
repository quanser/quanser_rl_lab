# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Derived from robotis_lab's OMY articulation configuration:
# Copyright 2025 ROBOTIS CO., LTD., Apache-2.0, author Taehyeong Kim.
# See THIRD_PARTY_LICENSES.md.

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from quanser_rl_lab.assets import QUANSER_RL_LAB_ASSETS_DATA_DIR

# Robot description lives in source/quanser_rl_lab/data/, tracked via Git LFS.
QARM_RESEARCH_USD = f"{QUANSER_RL_LAB_ASSETS_DATA_DIR}/robots/QArmResearch/qarm_research.usd"

# First bytes of a Git LFS pointer file, per the v1 spec.
_LFS_POINTER_MAGIC = b"version https://git-lfs"


def verify_usd_available() -> None:
    """Raise if the robot USD is missing or is an unfetched Git LFS pointer.

    Called when the environment cfg is built, not at import. tasks/__init__.py
    imports every task package, so raising here on import would stop unrelated
    robots' tasks from registering because of this one's missing asset.

    The pointer case is the common one: `git clone` without `git lfs pull`
    leaves a 133-byte text stub at the USD path, so os.path.isfile passes and
    the failure surfaces hundreds of lines into Isaac Sim startup as an
    unrelated-looking prim error.
    """
    if not os.path.isfile(QARM_RESEARCH_USD):
        # Two causes, and the path alone does not distinguish them, so name both.
        raise FileNotFoundError(
            f"QArm Research USD not found at {QARM_RESEARCH_USD}.\n"
            "Either the repository was cloned without Git LFS -- run "
            "`git lfs install && git lfs pull` from the repository root -- or the "
            "package was installed non-editable, which copies the package without "
            "its sibling data/ directory. Reinstall with `pip install -e`."
        )
    with open(QARM_RESEARCH_USD, "rb") as f:
        if f.read(len(_LFS_POINTER_MAGIC)) == _LFS_POINTER_MAGIC:
            raise FileNotFoundError(
                f"QArm Research USD at {QARM_RESEARCH_USD} is an unfetched Git LFS "
                "pointer, not the asset.\nRun `git lfs install && git lfs pull` from "
                "the repository root."
            )


# Joint and link names below (joint1..joint6, rh_*) are the prim names baked
# into the USD, so they stay as the asset spells them.
QARM_RESEARCH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=QARM_RESEARCH_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.260,
            "joint2": 0.869,
            "joint3": -2.000,
            "joint4": -0.0008,
            "joint5": -0.0008,
            "joint6": 0.260,
            "rh_r1_joint": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            velocity_limit_sim=0.8,
            effort_limit_sim=61.4,
            stiffness=80.0,
            damping=20.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-6]"],
            velocity_limit_sim=0.8,
            effort_limit_sim=31.7,
            stiffness=8.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"],
            velocity_limit_sim=1.0,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
    },
)
