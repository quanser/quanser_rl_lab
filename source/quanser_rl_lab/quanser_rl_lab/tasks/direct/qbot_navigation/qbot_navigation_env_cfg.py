# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""QBot Platform mapless-navigation task.

Conventions anything consuming a trained policy must match: action scaling,
control rate, lidar ray ordering, as documented in docs/qbot_mapless_nav/README.md.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from quanser_rl_lab.assets.qbot_platform import QBOT_CFG

# Physics timestep, in seconds
SIM_DT: float = 0.01

# Arena centre in world xy. Arena bounds, the goal marker and the robot's
# spawn pose are all expressed relative to this.
ARENA_CENTER: tuple[float, float] = (2.0, -3.0)


def cone_raycast_target() -> MultiMeshRayCasterCfg.RaycastTargetCfg:
    """Raycast target for the cone obstacles.

    A factory, not a constant: ArenaCfg and resync() must each hold their own
    instance or they alias, and resync() has to re-add it after a previous
    pass dropped it for num_obstacles == 0.
    """
    return MultiMeshRayCasterCfg.RaycastTargetCfg(
        prim_expr="/World/envs/env_.*/Venue/Cone.*",
        track_mesh_transforms=True,
    )


@configclass
class ArenaCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.315,
                dynamic_friction=0.315,
            ),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    qbot_platform = QBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/QBotPlatform")
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/QBotPlatform/base_link",
        history_length=1,
        track_air_time=False,
    )
    lidar_sensor = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/QBotPlatform/base_link",
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/defaultGroundPlane",
                is_shared=True,
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Venue/Wall.*",
                track_mesh_transforms=False,
            ),
            cone_raycast_target(),
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            # Placeholder; resync() overwrites it from num_lidar_rays.
            horizontal_res=360.0 / 42,
        ),
        ray_alignment="base",
        max_distance=8.0,
        debug_vis=False,
    )


@configclass
class RewardsCfg:
    """Reward terms, as plain signed values. Tuned, and validated over three
    seeds at 81-85% success.

    Once, on the terminating step:
        goal, collision, timeout
    Per metre of distance closed, on (d_{t-1} - d_t), signed:
        progress_scale
    Per control step:
        time_penalty    flat
        obstacle_scale  x max(0, obstacle_ramp_threshold - min(lidar scan))

    obstacle_ramp_threshold is a distance in metres. The two per-control-step
    terms need rescaling by hand if decimation changes; the others do not.
    """

    goal: float = 150.0
    collision: float = -250.0
    timeout: float = -100.0
    progress_scale: float = 300.0
    time_penalty: float = -0.5
    obstacle_scale: float = -1.0
    obstacle_ramp_threshold: float = 1.35


@configclass
class QBotNavEnvCfg(DirectRLEnvCfg):
    """Configuration for QBotNavEnv.

    The fields DirectRLEnvCfg requires: observation_space, action_space,
    scene, sim. They are derived from the ones below in __post_init__/resync().
    """

    # Arena / episode constants
    wall_height: float = 0.3
    wall_thickness: float = 0.1
    max_steps: int = 3000
    collision_force_thresh: float = 0.2
    # Physics steps ignored at episode start, while the robot settles from
    # its 0.1 m spawn.
    settle_steps: int = 20
    stiction_delay_steps: int = 36
    goal_reach_radius: float = 0.25
    goal_robot_min_dist: float = 1.5

    # Robot / physics
    left_damping: float = 0.209
    right_damping: float = 0.225
    left_stiffness: float = 0.0
    right_stiffness: float = 0.0
    ground_friction: float = 0.315
    wheel_radius: float = 3.5 * 0.0254 / 2
    wheel_base: float = 0.3928
    wheel_speed_limit: float = 15.75
    wheel_accel_limit: float = 11.25
    wheel_deadband: float = 0.3927
    wheel_decel_gain: float = 10.0

    # Wheel and yaw noise parameters, fitted using physical data
    wheel_noise_bias: float = 0.0289440579267636
    wheel_noise_white_std: float = 0.7873814624794425
    yaw_noise_bias: float = -0.0007551108151740715
    yaw_noise_white_std: float = 0.03288174469647006

    # Perturbs only the pose behind the goal observation; reward and
    # termination stay ground-truth. x and y share a std, drawn independently.
    localization_xy_noise_std: float = 0.025
    localization_yaw_noise_std: float = 0.006

    # Publish diagnostic keys in the step info dict and accumulate the
    # distances they need. Off by default.
    debug_extras: bool = False

    # Lidar
    lidar_max_range: float = 8.0
    lidar_range_std: float = 0.02
    lidar_dropout_prob: float = 0.02
    # Single source of truth for the ray count: resync() derives the pattern's
    # horizontal_res from it, and obs_dim = num_lidar_rays + 5.
    num_lidar_rays: int = 42

    rewards: RewardsCfg = RewardsCfg()

    # Policy and hardware at 10 Hz, sim at 100 Hz.
    decimation: int = 10
    # max_steps counts *physics* steps at SIM_DT, hence max_steps*SIM_DT.
    episode_length_s: float = max_steps * SIM_DT  # 30 s

    # Simulator / scene layout
    device: str = "cuda:0"
    num_envs: int = 4
    # Set with arena_width/height as a pair: parked cones live in the gap
    # between adjacent arenas, and resync enforces at least 2.0 m of it.
    env_spacing: float = 6.0
    ground_size: float = 0.0  # <=0 triggers auto-sizing
    arena_width: float = 4.0
    arena_height: float = 4.0
    num_obstacles: int = 9

    # Physical-unit scale AND hard clip for the policy action, applied in
    # QBotNavEnv._pre_physics_step. The actor bounds the Gaussian mean to (-1, 1).
    # Deterministic inference uses the mean, so hardware needs only the scale.
    # Both limits sit under the wheel ceiling (~0.70 m/s, ~3.56 rad/s).
    max_lin_vel: float = 0.6
    max_ang_vel: float = 2.4

    # num_obstacles is the fixed count of cone *prims* per env; the curriculum
    # varies how many are placed each reset and parks the rest outside the
    # walls (QBotNavEnv._park_position). Gated on success rate, not steps.
    curriculum_enabled: bool = True
    curriculum_start_obstacles: int = 2
    curriculum_success_threshold: float = 0.7
    curriculum_window: int = 500

    # Position sampling
    robot_obstacle_min_dist: float = 0.7
    obstacle_obstacle_min_dist: float = 0.8
    goal_obstacle_min_dist: float = 0.35
    max_position_sampling_tries: int = 200

    def __post_init__(self):
        # sim and scene are built here and *only* here: train.py and play.py
        # set scene.num_envs and sim.device on these objects directly, never on
        # the cfg fields of the same name, so rebuilding either in resync()
        # would silently discard --num_envs and --device.
        self.sim = SimulationCfg(dt=SIM_DT, device=self.device, render_interval=self.decimation)
        self.scene = ArenaCfg(num_envs=self.num_envs, env_spacing=self.env_spacing)
        self.resync()

    def resync(self) -> None:
        """Validate the tunable fields and re-derive everything computed from them.

        Called from __post_init__ and again from QBotNavEnv.__init__ before the
        scene is built. The second call is what makes post-construction
        overrides take effect, Hydra applies CLI overrides to an
        already-constructed cfg, and scripts assign fields the same way.
        Without it those fields move while everything derived from them does
        not, which fails silently rather than raising.
        """
        self.episode_length_s = self.max_steps * SIM_DT

        # --- validation scene parameters ---
        if self.max_lin_vel <= 0.0 or self.max_ang_vel <= 0.0:
            raise ValueError("max_lin_vel and max_ang_vel must be > 0.")
        if self.wheel_speed_limit <= 0.0:
            raise ValueError("wheel_speed_limit must be > 0.")
        if self.wheel_accel_limit <= 0.0:
            raise ValueError("wheel_accel_limit must be > 0.")
        if self.wheel_decel_gain <= 0.0:
            raise ValueError("wheel_decel_gain must be > 0.")

        if not 0.0 < self.curriculum_success_threshold <= 1.0:
            raise ValueError("curriculum_success_threshold must be in (0, 1].")
        # Clamped, not rejected: overriding num_obstacles down (0 is supported)
        self.curriculum_start_obstacles = min(self.curriculum_start_obstacles, self.num_obstacles)

        # The geometry contracts behind QBotNavEnv._park_position:
        # an over-long row overlaps held-out cones, which are dynamic
        # bodies the solver flings apart, and too little spacing puts them in
        # the neighbour's arena as phantom lidar obstacles.
        if self.curriculum_enabled and self.num_obstacles > 0:
            if self.num_obstacles * 0.4 > self.arena_width:
                raise ValueError(
                    "parked-cone row (num_obstacles * 0.4 m) does not fit along arena_width;"
                    " reduce num_obstacles or widen the arena."
                )
            if self.env_spacing - max(self.arena_width, self.arena_height) < 2.0:
                raise ValueError(
                    "env_spacing leaves under 2.0 m between adjacent arenas, so parked cones"
                    " would sit inside a neighbouring env's arena."
                )

        # --- observation parameters ---
        self.wheel_speed_obs_max = self.wheel_speed_limit
        self.gyro_yaw_rate_obs_max = self.max_ang_vel
        self.obs_dim = self.num_lidar_rays + 5

        # --- arena bounds, from ARENA_CENTER ---
        _cx, _cy = ARENA_CENTER
        self.arena_x_min = _cx - self.arena_width / 2.0
        self.arena_x_max = _cx + self.arena_width / 2.0
        self.arena_y_min = _cy - self.arena_height / 2.0
        self.arena_y_max = _cy + self.arena_height / 2.0

        # --- DirectRLEnvCfg required fields ---
        self.observation_space = self.obs_dim
        self.action_space = 2
        self.state_space = 0  # no privileged critic state; 0 avoids None serialization error in Hydra

        self.sim.render_interval = self.decimation

        # --- scene, edited in place; __post_init__ owns num_envs and device ---
        scene = self.scene
        scene.env_spacing = self.env_spacing
        _gs = self._auto_ground_size()
        scene.ground.spawn.size = (_gs, _gs)
        scene.ground.spawn.physics_material.static_friction = self.ground_friction
        scene.ground.spawn.physics_material.dynamic_friction = self.ground_friction
        # Asset defaults live in qbot_platform.py; these overrides are what
        # make per-run gain changes from the CLI actually reach training.
        scene.qbot_platform.actuators["left_wheel"].damping = self.left_damping
        scene.qbot_platform.actuators["right_wheel"].damping = self.right_damping
        scene.qbot_platform.actuators["left_wheel"].stiffness = self.left_stiffness
        scene.qbot_platform.actuators["right_wheel"].stiffness = self.right_stiffness
        # Rebuilt from the non-cone targets rather than filtered in place, so
        # it can restore the cone target as well as drop it.
        targets = [t for t in scene.lidar_sensor.mesh_prim_paths if "Cone" not in t.prim_expr]
        if self.num_obstacles > 0:
            targets.append(cone_raycast_target())
        scene.lidar_sensor.mesh_prim_paths = targets
        scene.lidar_sensor.max_distance = self.lidar_max_range
        scene.lidar_sensor.pattern_cfg.horizontal_res = 360.0 / self.num_lidar_rays

    def _auto_ground_size(self) -> float:
        if self.ground_size > 0.0:
            return self.ground_size
        grid_side = int(math.ceil(math.sqrt(max(1, self.scene.num_envs))))
        venue_extent = max(self.arena_width, self.arena_height)
        return max(100.0, grid_side * self.env_spacing + venue_extent + 4.0)


@configclass
class QBotNavEnvCfg_PLAY(QBotNavEnvCfg):
    """Evaluation variant: full obstacle density from the first episode.

    The curriculum advances only during training and its level is not stored
    in the checkpoint, so replaying a 9-cone policy through the training cfg
    would evaluate it at the 2-cone starting level.
    """

    def __post_init__(self):
        self.curriculum_enabled = False
        super().__post_init__()
