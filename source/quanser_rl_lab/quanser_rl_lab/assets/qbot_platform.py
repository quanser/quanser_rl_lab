# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from quanser_rl_lab.assets import QUANSER_RL_LAB_ASSETS_DATA_DIR

# Robot description lives in source/quanser_rl_lab/data/, tracked via Git LFS.
QBOT_PLATFORM_USD = f"{QUANSER_RL_LAB_ASSETS_DATA_DIR}/robots/QBotPlatform/qbot_platform.usd"

# First bytes of a Git LFS pointer file, per the v1 spec.
_LFS_POINTER_MAGIC = b"version https://git-lfs"


def verify_usd_available() -> None:
    """Raise if the robot USD is missing or is an unfetched Git LFS pointer.

    Called when the environment is built, not at import. tasks/__init__.py
    imports every task package, so raising here on import would stop unrelated
    robots' tasks from registering because of this one's missing asset.

    The pointer case is the common one: `git clone` without `git lfs pull`
    leaves a 133-byte text stub at the USD path, so os.path.isfile passes and
    the failure surfaces hundreds of lines into Isaac Sim startup as an
    unrelated-looking prim error.
    """
    if not os.path.isfile(QBOT_PLATFORM_USD):
        # Two causes, and the path alone does not distinguish them, so name both.
        raise FileNotFoundError(
            f"QBot Platform USD not found at {QBOT_PLATFORM_USD}.\n"
            "Either the repository was cloned without Git LFS -- run "
            "`git lfs install && git lfs pull` from the repository root -- or the "
            "package was installed non-editable, which copies the package without "
            "its sibling data/ directory. Reinstall with `pip install -e`."
        )
    with open(QBOT_PLATFORM_USD, "rb") as f:
        if f.read(len(_LFS_POINTER_MAGIC)) == _LFS_POINTER_MAGIC:
            raise FileNotFoundError(
                f"QBot Platform USD at {QBOT_PLATFORM_USD} is an unfetched Git LFS "
                "pointer, not the asset.\nRun `git lfs install && git lfs pull` from "
                "the repository root."
            )


QBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=QBOT_PLATFORM_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.0, -3.0, 0.1),
        joint_pos={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0},
    ),
    actuators={
        "left_wheel": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint"],
            effort_limit_sim=3.4e38,
            velocity_limit_sim=float("inf"),
            stiffness=0.0,
            damping=0.1,
        ),
        "right_wheel": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint"],
            effort_limit_sim=3.4e38,
            velocity_limit_sim=float("inf"),
            stiffness=0.0,
            damping=0.1,
        ),
    },
)
