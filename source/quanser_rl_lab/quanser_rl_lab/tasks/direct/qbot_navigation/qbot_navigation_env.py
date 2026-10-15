# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""QBot Platform mapless-navigation task.

Conventions anything consuming a trained policy must match: action scaling,
control rate, lidar ray ordering, as documented in docs/qbot_mapless_nav/README.md.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import omni.log
import omni.usd
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.utils.noise import GaussianNoiseCfg, gaussian_noise

from quanser_rl_lab.actuators.stiction_filter import StictionFilter
from quanser_rl_lab.assets.qbot_platform import verify_usd_available

from .qbot_navigation_env_cfg import ARENA_CENTER, QBotNavEnvCfg


class QBotNavEnv(DirectRLEnv):
    """DirectRLEnv implementation of the QBot Platform mapless navigation task."""

    cfg: QBotNavEnvCfg

    def __init__(self, cfg: QBotNavEnvCfg, render_mode: str | None = None, **kwargs):
        # The config is final at this point and not before: Hydra and the
        # scripts assign to an already-constructed cfg. Must precede
        # super().__init__, which builds the scene from cfg.scene.
        cfg.resync()
        # Before the scene spawns the robot, so a missing or unfetched-LFS USD
        # reports itself rather than surfacing as an Isaac Sim prim error.
        verify_usd_available()
        super().__init__(cfg, render_mode, **kwargs)

        self.qbot: Articulation = self.scene["qbot_platform"]
        self.contact: ContactSensor = self.scene["contact_sensor"]
        self.lidar: RayCaster = self.scene["lidar_sensor"]

        self._env_origins_t = self.scene.env_origins[:, :2].clone().to(self.device)

        self.stiction = StictionFilter(
            num_envs=self.num_envs,
            num_wheels=2,
            delay_steps=cfg.stiction_delay_steps,
            device=self.device,
        )
        self._wheel_noise_cfg = GaussianNoiseCfg(
            mean=cfg.wheel_noise_bias, std=cfg.wheel_noise_white_std, operation="add"
        )
        self._yaw_noise_cfg = GaussianNoiseCfg(mean=cfg.yaw_noise_bias, std=cfg.yaw_noise_white_std, operation="add")
        # Applied to the goal observation's pose only (_get_goal_polar);
        self._localization_xy_noise_cfg = GaussianNoiseCfg(mean=0.0, std=cfg.localization_xy_noise_std, operation="add")
        self._localization_yaw_noise_cfg = GaussianNoiseCfg(
            mean=0.0, std=cfg.localization_yaw_noise_std, operation="add"
        )

        N = self.num_envs
        self._goal_positions = torch.zeros(N, 2, device=self.device)
        self._prev_dist = torch.zeros(N, device=self.device)
        self._episode_distance_traveled = torch.zeros(N, device=self.device)
        self._episode_progress_reward = torch.zeros(N, device=self.device)
        # Per-episode accumulators, for extras["log"] only (see _reset_idx).
        self._episode_collision_steps = torch.zeros(N, device=self.device)
        self._episode_obstacle_penalty = torch.zeros(N, device=self.device)
        self._episode_time_penalty = torch.zeros(N, device=self.device)
        self._obs_noisy_wheel = torch.zeros(N, 2, device=self.device)
        self._obs_noisy_gyro = torch.zeros(N, device=self.device)

        # Cached between _get_dones() and _get_rewards() within one step.
        self._curr_dist = torch.zeros(N, device=self.device)
        self._active = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._collision = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._goal_reached = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Raw sensor readings in physical units, drawn once per step by
        # _update_obs_cache and consumed by _build_obs.
        self._lidar_scan = torch.zeros(N, cfg.num_lidar_rays, device=self.device)
        self._obs_goal_polar = torch.zeros(N, 2, device=self.device)
        # The assembled, normalised vector the two above feed into.
        self._cached_obs = torch.zeros(N, cfg.obs_dim, device=self.device)

        # Collisions seen across the decimation sub-steps, accumulated in
        # _apply_action and OR-ed with the final reading in _get_dones.
        self._substep_collision = torch.zeros(N, dtype=torch.bool, device=self.device)

        self._actions = torch.zeros(N, 2, device=self.device)
        # Scale and hard clip for the action.
        self._action_clip_bound = torch.tensor([cfg.max_lin_vel, cfg.max_ang_vel], device=self.device)
        self._prev_wheel_cmd = torch.zeros(N, 2, device=self.device)

        self._robot_yaw = np.zeros(N, dtype=np.float32)

        # Cones placed inside the arena right now; the rest are parked outside
        # the walls each reset. Ramps up in _update_curriculum.
        self._active_num_obstacles = cfg.curriculum_start_obstacles if cfg.curriculum_enabled else cfg.num_obstacles
        # Recent episode outcomes (1.0 = reached goal) as a ring buffer, O(1)
        # per episode. _curriculum_filled counts how much of it is real data.
        self._curriculum_outcomes = torch.zeros(cfg.curriculum_window, device=self.device)
        self._curriculum_cursor = 0
        self._curriculum_filled = 0

        # settle_steps is defined in physics steps (the cfg's SIM_DT).
        # episode_length_buf counts policy steps, so divide by decimation.
        self._settle_policy_steps = max(1, cfg.settle_steps // self.cfg.decimation)

    # --- Scene setup ---

    def _setup_scene(self):
        """Called by base __init__ after InteractiveScene is built, before sim.reset()."""
        self._spawn_venue()
        self._spawn_goal_markers()

    # --- Action pipeline ---

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Actions arrive centred on a tanh-bounded mean but are not bounded
        # themselves (std is capped at 0.3, not zero), so scale to physical
        # [lin_vel, ang_vel] units and clamp to the same bound.
        self._actions = (actions * self._action_clip_bound).clamp(-self._action_clip_bound, self._action_clip_bound)
        self._substep_collision.fill_(False)

    def _process_wheel_command(self, wheel_cmd: torch.Tensor) -> torch.Tensor:
        """Apply hardware-style deadband, asymmetric accel limits, and wheel cap."""

        cmd = wheel_cmd.clamp(-self.cfg.wheel_speed_limit, self.cfg.wheel_speed_limit)

        if self.cfg.wheel_deadband > 0.0:
            deadband_mask = cmd.abs() < self.cfg.wheel_deadband
            cmd = torch.where(deadband_mask, torch.zeros_like(cmd), cmd)

        accel = self.cfg.wheel_accel_limit
        decel = self.cfg.wheel_decel_gain * accel
        upper_rate = torch.where(
            cmd > 0.0,
            torch.full_like(cmd, accel),
            torch.where(cmd < 0.0, torch.full_like(cmd, decel), torch.full_like(cmd, accel)),
        )
        lower_rate = torch.where(
            cmd > 0.0,
            torch.full_like(cmd, -decel),
            torch.where(cmd < 0.0, torch.full_like(cmd, -accel), torch.full_like(cmd, -accel)),
        )

        max_up = upper_rate * self.physics_dt
        max_down = lower_rate * self.physics_dt
        delta = (cmd - self._prev_wheel_cmd).clamp(min=max_down, max=max_up)
        cmd_processed = self._prev_wheel_cmd + delta
        cmd_processed = cmd_processed.clamp(-self.cfg.wheel_speed_limit, self.cfg.wheel_speed_limit)

        self._prev_wheel_cmd.copy_(cmd_processed)
        return cmd_processed

    def _apply_action(self) -> None:
        wheel_vel = self.qbot.data.joint_vel[:, :2]
        ang_vel_w = self.qbot.data.root_ang_vel_w[:, 2]

        # Body commands [linear_x_mps, angular_z_radps] -> wheel commands,
        # via differential-drive inverse kinematics.
        lin_vel = self._actions[:, 0]
        ang_vel = self._actions[:, 1]
        left = (lin_vel - 0.5 * self.cfg.wheel_base * ang_vel) / self.cfg.wheel_radius
        right = (lin_vel + 0.5 * self.cfg.wheel_base * ang_vel) / self.cfg.wheel_radius
        wheel_cmd = torch.stack([left, right], dim=1)
        wheel_cmd = self._process_wheel_command(wheel_cmd)
        filtered = self.stiction.apply(wheel_cmd, wheel_vel)
        self.qbot.set_joint_velocity_target(filtered)

        lin_vel_gt = (wheel_vel[:, 0] + wheel_vel[:, 1]) * 0.5 * self.cfg.wheel_radius
        if self.cfg.debug_extras:
            self._episode_distance_traveled.add_(lin_vel_gt.abs() * self.physics_dt)

        # _get_dones ORs this with the final sub-step's reading.
        forces = self.contact.data.net_forces_w
        max_f = torch.norm(forces, dim=-1).amax(dim=1)
        self._substep_collision |= max_f > self.cfg.collision_force_thresh

        # Plain rebinds, not accumulators: only the final substep's draw
        # reaches the observation and the other decimation-1 draws are
        # discarded. Moving this to _update_obs_cache would sample one physics
        # step later.
        self._obs_noisy_wheel = gaussian_noise(wheel_vel, self._wheel_noise_cfg)
        self._obs_noisy_gyro = gaussian_noise(ang_vel_w, self._yaw_noise_cfg)

    # --- Dones  (called first in step, before _get_rewards and _reset_idx) ---

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Exactly once per step, before _build_obs below. _update_obs_cache's
        # docstring says why this cannot live in _get_observations.
        self._update_obs_cache()

        active = self.episode_length_buf >= self._settle_policy_steps

        pos_w = self.qbot.data.root_pos_w
        pos_local = pos_w[:, :2] - self._env_origins_t

        forces = self.contact.data.net_forces_w
        max_f = torch.norm(forces, dim=-1).amax(dim=1)
        curr_dist = torch.sqrt(
            (pos_local[:, 0] - self._goal_positions[:, 0]) ** 2 + (pos_local[:, 1] - self._goal_positions[:, 1]) ** 2
        )

        collision = active & (self._substep_collision | (max_f > self.cfg.collision_force_thresh))
        # Distance only, no velocity condition, deliberately: the goal radius
        # is a handover point and the robot drives through it at a median
        # 0.23 m/s. A speed gate would also fight the time penalty and the
        # action noise std, which is global rather than a network output.
        goal_reached = active & (curr_dist < self.cfg.goal_reach_radius)
        truncated = self.episode_length_buf >= self.max_episode_length
        # Collision terminates: as a repeated per-step penalty it let an env
        # stay wedged, producing heavy-tailed returns that destabilized PPO.
        terminated = goal_reached | collision

        self._curr_dist = curr_dist
        self._active = active
        self._collision = collision
        self._goal_reached = goal_reached

        # Pre-reset state; _get_observations reuses it on steps with no
        # resets, so _build_obs runs once per step in the common case.
        self._cached_obs = self._build_obs()

        # time_outs is required: rsl_rl reads it to bootstrap truncated
        # episodes. The rest is diagnostic -- see cfg.debug_extras.
        self.extras["time_outs"] = truncated
        if self.cfg.debug_extras:
            self.extras["final_obs"] = self._cached_obs
            self.extras["goal_reached"] = goal_reached
            self.extras["collision"] = collision
            self.extras["timeout"] = truncated
            self.extras["episode_distance_traveled"] = self._episode_distance_traveled.clone()
            self.extras["episode_distance_progress_reward"] = self._episode_progress_reward.clone()

        return terminated, truncated

    # --- Rewards  (called after _get_dones, before _reset_idx) ---

    def _get_rewards(self) -> torch.Tensor:
        r = self._compute_rewards()
        return torch.nan_to_num(r, nan=0.0)

    def _compute_rewards(self) -> torch.Tensor:
        active = self._active
        curr_dist = self._curr_dist

        reward = torch.zeros(self.num_envs, device=self.device)
        reward[active] += self.cfg.rewards.time_penalty
        self._episode_time_penalty[active] += self.cfg.rewards.time_penalty

        # Signed, not clamped at 0: clamping lets the agent farm reward by
        # reversing away for free.
        progress = self.cfg.rewards.progress_scale * (self._prev_dist - curr_dist)
        reward[active] += progress[active]

        obs_penalty = self._obstacle_proximity_penalty()
        reward[active] += obs_penalty[active]

        # At most once per episode: a collision also terminates it.
        reward[self._collision] += self.cfg.rewards.collision
        reward[self._goal_reached] += self.cfg.rewards.goal
        # Masked to exclude envs that also terminated this step, so a
        # coincidence is not double-penalized.
        timeout_only = self.reset_time_outs & ~self._goal_reached & ~self._collision
        reward[timeout_only] += self.cfg.rewards.timeout

        self._episode_progress_reward[active] += progress[active]
        self._episode_collision_steps[active] += self._collision[active].float()
        self._episode_obstacle_penalty[active] += obs_penalty[active]
        self._prev_dist = curr_dist.clone()
        return reward

    def _obstacle_proximity_penalty(self) -> torch.Tensor:
        """
        Computes the penalty directly from self._lidar_scan, using the same
        scan sampled for the policy observation. This ensures that the policy
        and reward function operate on consistent sensor data.

        Using LiDAR measurements instead of explicit obstacle poses also
        allows the penalty to account for arbitrary obstacles, including
        walls, without special-casing their geometry.
        """
        min_scan = self._lidar_scan.min(dim=1).values
        intrusion = torch.clamp(self.cfg.rewards.obstacle_ramp_threshold - min_scan, min=0.0)
        return self.cfg.rewards.obstacle_scale * intrusion

    # --- Observations  (called after _reset_idx -- returns post-reset obs) ---

    def _get_observations(self) -> dict:
        # With no resets the scene is what _get_dones already observed. With
        # resets, _reset_idx changed pose and sensor state for the done envs,
        # so the observation has to be rebuilt.
        if not self.reset_buf.any():
            return {"policy": self._cached_obs}
        return {"policy": self._build_obs()}

    def _update_obs_cache(self) -> None:
        """Recompute this step's sensor values into the per-step cache.

        Called once per step from the top of _get_dones(), not from
        _get_observations(): the latter only rebuilds for steps with resets,
        so drawing there would give resetting envs an extra round of sensor
        noise that the others did not get in the same step.

        _compute_lidar_obs draws fresh range noise and dropout every call, so
        caching is what makes the obstacle reward score the same scan the
        policy saw. Wheel and gyro are not refreshed here, _apply_action
        overwrites them once per substep and the last one stands.
        """
        self._lidar_scan = self._compute_lidar_obs()
        self._obs_goal_polar = self._get_goal_polar()

    def _build_obs(self) -> torch.Tensor:
        lidar = self._lidar_scan / self.cfg.lidar_max_range
        goal = self._obs_goal_polar.clone()
        goal[:, 0] /= self.cfg.lidar_max_range
        goal[:, 1] /= math.pi
        obs_parts = [lidar, goal]
        proprio = torch.cat([self._obs_noisy_wheel, self._obs_noisy_gyro.unsqueeze(1)], dim=1)
        proprio[:, :2] = proprio[:, :2] / self.cfg.wheel_speed_obs_max
        proprio[:, 2] = proprio[:, 2] / self.cfg.gyro_yaw_rate_obs_max
        proprio = torch.clamp(proprio, -1.0, 1.0)
        obs_parts.append(proprio)
        obs = torch.cat(obs_parts, dim=1)
        return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _get_goal_polar(self) -> torch.Tensor:
        """Noisy-localization [dist_m, heading_rad] to the goal in the body
        frame, (N, 2). Observation only: _get_dones computes its own
        ground-truth distance for reward and termination."""
        pos_w = self.qbot.data.root_pos_w
        q = self.qbot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
        )
        yaw = gaussian_noise(yaw, self._localization_yaw_noise_cfg)
        pos_local = pos_w[:, :2] - self._env_origins_t
        pos_local = gaussian_noise(pos_local, self._localization_xy_noise_cfg)
        dx_w = self._goal_positions[:, 0] - pos_local[:, 0]
        dy_w = self._goal_positions[:, 1] - pos_local[:, 1]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        dx_b = cos_y * dx_w + sin_y * dy_w
        dy_b = sin_y * dx_w - cos_y * dy_w
        dist = torch.sqrt(dx_b**2 + dy_b**2).clamp(min=1e-4)
        heading = torch.atan2(dy_b, dx_b)
        return torch.stack([dist, heading], dim=1)

    def _compute_lidar_obs(self) -> torch.Tensor:
        """Vectorised lidar distances with noise. (N, NUM_LIDAR_RAYS)."""

        # Range only, from the world-frame hit points: rotations preserve the
        # norm, so no body-frame transform is needed.
        hits_w = self.lidar.data.ray_hits_w
        sensor_pos = self.lidar.data.pos_w
        dx_w = hits_w[:, :, 0] - sensor_pos[:, 0:1]
        dy_w = hits_w[:, :, 1] - sensor_pos[:, 1:2]
        dist = torch.sqrt(dx_w**2 + dy_w**2)
        dist = torch.nan_to_num(dist, nan=self.cfg.lidar_max_range, posinf=self.cfg.lidar_max_range)
        dist = dist + torch.randn_like(dist) * self.cfg.lidar_range_std
        dropout = torch.rand_like(dist) < self.cfg.lidar_dropout_prob
        dist = torch.where(dropout, torch.full_like(dist, self.cfg.lidar_max_range), dist)
        return dist.clamp(0.0, self.cfg.lidar_max_range)

    # --- Reset ---

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)

        env_ids_np = env_ids.cpu().numpy().astype(np.int64)

        # Before _sample_layouts_for_envs, which reads _active_num_obstacles.
        curriculum_success_rate = self._update_curriculum(env_ids)

        sampled_layouts = self._sample_layouts_for_envs(env_ids_np)
        self._reset_obstacle_envs(env_ids_np, sampled_layouts)
        self._apply_goal_layout(env_ids_np, sampled_layouts)
        self._reset_robot_envs(env_ids_np, sampled_layouts)

        self.stiction.reset(env_ids=env_ids)
        self._prev_wheel_cmd[env_ids] = 0.0

        self._obs_noisy_wheel[env_ids] = 0.0
        self._obs_noisy_gyro[env_ids] = 0.0

        # Per-episode diagnostics for the envs ending this step. Each "_total"
        # is that term summed over the episode, read before the accumulators
        # are zeroed below; the six reconstruct rsl_rl's "Mean reward".
        collision_penalty_total = self._episode_collision_steps[env_ids] * self.cfg.rewards.collision
        goal_bonus_total = self._goal_reached[env_ids].float() * self.cfg.rewards.goal
        time_penalty_total = self._episode_time_penalty[env_ids]
        progress_total = self._episode_progress_reward[env_ids]
        obstacle_penalty_total = self._episode_obstacle_penalty[env_ids]
        # Mutually exclusive, sums to 1.0. Priority goal > collision > timeout
        # for the rare step where two coincide. reset_time_outs is the base
        # class's truncated flag, already set before _reset_idx runs.
        is_goal_term = self._goal_reached[env_ids]
        is_collision_term = self._collision[env_ids] & ~is_goal_term
        is_timeout_term = self.reset_time_outs[env_ids] & ~is_goal_term & ~is_collision_term
        timeout_penalty_total = is_timeout_term.float() * self.cfg.rewards.timeout
        log = {
            "Episode_Termination/goal_reached": is_goal_term.float().mean().item(),
            "Episode_Termination/collision": is_collision_term.float().mean().item(),
            "Episode_Termination/timeout": is_timeout_term.float().mean().item(),
            "Curriculum/active_num_obstacles": float(self._active_num_obstacles),
            "Curriculum/success_rate_window": curriculum_success_rate,
            "Episode_Reward/time_penalty_total": time_penalty_total.mean().item(),
            "Episode_Reward/progress_total": progress_total.mean().item(),
            "Episode_Reward/obstacle_penalty_total": obstacle_penalty_total.mean().item(),
            "Episode_Reward/collision_penalty_total": collision_penalty_total.mean().item(),
            "Episode_Reward/goal_bonus_total": goal_bonus_total.mean().item(),
            "Episode_Reward/timeout_penalty_total": timeout_penalty_total.mean().item(),
        }
        self.extras["log"] = log

        self._episode_distance_traveled[env_ids] = 0.0
        self._episode_progress_reward[env_ids] = 0.0
        self._episode_collision_steps[env_ids] = 0.0
        self._episode_obstacle_penalty[env_ids] = 0.0
        self._episode_time_penalty[env_ids] = 0.0

        # Seed _prev_dist so the first progress reward is correct.
        for env_idx in env_ids_np:
            rx, ry = sampled_layouts[int(env_idx)]["robot"]
            gx, gy = sampled_layouts[int(env_idx)]["goal"]
            self._prev_dist[env_idx] = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)

        # --- Reseed observation cache for reset envs (must run last) ---
        # Goal geometry is computed analytically because root_pos_w remains stale
        # until the next physics step. LiDAR cannot bypass this delay.
        # Resample noise to keep post-reset observations consistent with normal steps.
        xy_std = self.cfg.localization_xy_noise_std
        yaw_std = self.cfg.localization_yaw_noise_std
        goal_seed_np = np.zeros((len(env_ids_np), 2), dtype=np.float32)
        for local_i, env_idx in enumerate(env_ids_np):
            rx, ry = sampled_layouts[int(env_idx)]["robot"]
            gx, gy = sampled_layouts[int(env_idx)]["goal"]
            rx_noisy = rx + np.random.normal(0.0, xy_std)
            ry_noisy = ry + np.random.normal(0.0, xy_std)
            yaw = float(self._robot_yaw[env_idx]) + np.random.normal(0.0, yaw_std)
            dx_w, dy_w = gx - rx_noisy, gy - ry_noisy
            # Same mirrored handedness as _get_goal_polar; a mismatch makes
            # the first observation after a reset disagree with the rest.
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            dx_b = cos_y * dx_w + sin_y * dy_w
            dy_b = sin_y * dx_w - cos_y * dy_w
            goal_seed_np[local_i, 0] = max(math.sqrt(dx_b**2 + dy_b**2), 1e-4)
            goal_seed_np[local_i, 1] = math.atan2(dy_b, dx_b)
        goal_seed = torch.as_tensor(goal_seed_np, device=self.device)

        # Overwrite the cache for the reset envs so _get_observations serves
        # their new state. Wheel and gyro were already zeroed above.
        self._obs_goal_polar[env_ids] = goal_seed
        self._lidar_scan[env_ids] = self._compute_lidar_obs()[env_ids]

    # --- Layout sampling ---

    def _sample_with_fallback(self, sample_fn, checks: list, label: str) -> tuple[float, float]:
        """Sample a position, progressively relaxing constraints if necessary.

        Attempts up to cfg.max_position_sampling_tries per constraint level.
        Constraints are ordered by priority (highest first) and dropped from
        last to first until a valid sample is found.

        Falls back to the last sampled position if all attempts fail, avoiding
        reset failures when the feasible region is too restrictive.
        """
        last = None
        for num_checks in range(len(checks), -1, -1):
            active_checks = checks[:num_checks]
            for _ in range(self.cfg.max_position_sampling_tries):
                x, y = sample_fn()
                last = (x, y)
                if all(check(x, y) for check in active_checks):
                    if num_checks < len(checks):
                        omni.log.warn(
                            f"[QBotNavEnv] relaxed constraints sampling {label} "
                            f"(dropped {len(checks) - num_checks}/{len(checks)} check(s))."
                        )
                    return x, y
        omni.log.warn(f"[QBotNavEnv] failed to sample {label} even fully relaxed; using last draw {last}.")
        return last

    def _sample_layouts_for_envs(self, env_ids: np.ndarray) -> dict:
        """Sample randomized robot, obstacle, and goal layouts for each environment.

        Positions respect arena margins and minimum separation distances.
        Sampling constraints are progressively relaxed in priority order
        when valid positions cannot be found.
        """
        layouts = {}
        for env_idx in env_ids:
            # Robot position only requires an arena margin.
            rx, ry = self._random_pos(margin=0.8)

            obstacle_positions = []
            for _ in range(self._active_num_obstacles):
                ox, oy = self._sample_with_fallback(
                    lambda: self._random_pos(margin=0.5),
                    [
                        lambda x, y: (
                            not any(
                                (x - px) ** 2 + (y - py) ** 2 < self.cfg.obstacle_obstacle_min_dist**2
                                for px, py in obstacle_positions
                            )
                        ),
                        lambda x, y: (x - rx) ** 2 + (y - ry) ** 2 >= self.cfg.robot_obstacle_min_dist**2,
                    ],
                    "obstacle position",
                )
                obstacle_positions.append((ox, oy))

            gx, gy = self._sample_with_fallback(
                lambda: self._random_pos(margin=0.5),
                [
                    # Goal clearance is independent of obstacle spacing.
                    lambda x, y: (
                        not any(
                            (x - px) ** 2 + (y - py) ** 2 < self.cfg.goal_obstacle_min_dist**2
                            for px, py in obstacle_positions
                        )
                    ),
                    lambda x, y: (x - rx) ** 2 + (y - ry) ** 2 >= self.cfg.goal_robot_min_dist**2,
                ],
                "goal position",
            )

            layouts[int(env_idx)] = {
                "robot": (rx, ry),
                "obstacles": obstacle_positions,
                "goal": (gx, gy),
            }
        return layouts

    def _park_position(self, cone_idx: int) -> tuple[float, float]:
        """Return an env-local parking position for an inactive obstacle.

        Park cones 1 m beyond the south wall, outside LiDAR visibility but
        within the gap between arenas to avoid affecting neighboring envs.
        Space cones 0.4 m apart to prevent physics solver interference.

        Parking capacity and arena spacing are validated in resync().
        """
        return (
            self.cfg.arena_x_min + 0.4 * cone_idx,
            self.cfg.arena_y_min - 1.0,
        )

    def _update_curriculum(self, env_ids: torch.Tensor) -> float:
        """Update the rolling success rate and increase difficulty when eligible.

        Called before layout sampling so level-ups apply to the next episodes.
        Returns the success rate for logging.
        """
        if not self.cfg.curriculum_enabled:
            return 0.0

        outcomes = self._goal_reached[env_ids].float()
        n = int(outcomes.numel())
        window = self.cfg.curriculum_window
        if n == 0:
            return 0.0

        # Ring buffer; retain only the latest outcomes if n exceeds window.
        if n >= window:
            self._curriculum_outcomes.copy_(outcomes[-window:])
            self._curriculum_cursor = 0
            self._curriculum_filled = window
        else:
            end = self._curriculum_cursor + n
            if end <= window:
                self._curriculum_outcomes[self._curriculum_cursor : end] = outcomes
            else:
                split = window - self._curriculum_cursor
                self._curriculum_outcomes[self._curriculum_cursor :] = outcomes[:split]
                self._curriculum_outcomes[: end - window] = outcomes[split:]
            self._curriculum_cursor = end % window
            self._curriculum_filled = min(window, self._curriculum_filled + n)

        if self._curriculum_filled < window:
            # Report partial success rate without triggering a level-up.
            return float(self._curriculum_outcomes[: self._curriculum_filled].mean().item())

        success_rate = float(self._curriculum_outcomes.mean().item())
        if (
            success_rate >= self.cfg.curriculum_success_threshold
            and self._active_num_obstacles < self.cfg.num_obstacles
        ):
            self._active_num_obstacles += 1
            # Reset history to prevent repeated level-ups from stale outcomes.
            self._curriculum_outcomes.zero_()
            self._curriculum_cursor = 0
            self._curriculum_filled = 0
        return success_rate

    def _random_pos(self, margin: float = 0.8) -> tuple[float, float]:
        return (
            np.random.uniform(self.cfg.arena_x_min + margin, self.cfg.arena_x_max - margin),
            np.random.uniform(self.cfg.arena_y_min + margin, self.cfg.arena_y_max - margin),
        )

    # --- Per-env state reset helpers ---

    def _reset_robot_envs(self, env_ids: np.ndarray, sampled_layouts: dict) -> None:
        """Reset selected robots to their sampled positions with random headings.

        Convert env-local positions to world coordinates, then reset root
        velocities, joint positions, and joint velocities to zero.
        """
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        root_pose = self.qbot.data.default_root_state[env_ids_t, :7].clone()
        root_vel = self.qbot.data.default_root_state[env_ids_t, 7:].clone()
        root_vel[:] = 0.0

        for local_i, env_idx in enumerate(env_ids):
            x, y = sampled_layouts[int(env_idx)]["robot"]
            yaw = np.random.uniform(0, 2 * math.pi)
            self._robot_yaw[env_idx] = yaw
            origin = self.scene.env_origins[env_idx]
            root_pose[local_i, 0] = float(x) + float(origin[0])
            root_pose[local_i, 1] = float(y) + float(origin[1])
            root_pose[local_i, 2] = 0.1 + float(origin[2])
            root_pose[local_i, 3] = float(math.cos(float(yaw) / 2.0))
            root_pose[local_i, 4] = 0.0
            root_pose[local_i, 5] = 0.0
            root_pose[local_i, 6] = float(math.sin(float(yaw) / 2.0))

        self.qbot.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)
        self.qbot.write_root_velocity_to_sim(root_vel, env_ids=env_ids_t)
        j_pos = self.qbot.data.default_joint_pos[env_ids_t].clone()
        j_vel = self.qbot.data.default_joint_vel[env_ids_t].clone()
        j_pos[:] = 0.0
        j_vel[:] = 0.0
        self.qbot.write_joint_state_to_sim(j_pos, j_vel, env_ids=env_ids_t)
        self.qbot.reset(env_ids=env_ids_t)

    def _reset_obstacle_envs(self, env_ids: np.ndarray, sampled_layouts: dict) -> None:
        """Reset obstacles to sampled positions, parking inactive ones outside the arena.

        Convert env-local positions to world coordinates, randomize headings,
        and clear root velocities.
        """
        if len(env_ids) == 0 or not self.cone_objects:
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for cone_idx, cone_obj in enumerate(self.cone_objects):
            root_pose = cone_obj.data.default_root_state[env_ids_t, :7].clone()
            root_vel = cone_obj.data.default_root_state[env_ids_t, 7:].clone()
            root_vel[:] = 0.0
            for local_i, env_idx in enumerate(env_ids):
                obs_list = sampled_layouts[int(env_idx)]["obstacles"]
                if cone_idx < len(obs_list):
                    x, y = obs_list[cone_idx]
                else:
                    # Park inactive obstacles to preserve curriculum difficulty.
                    x, y = self._park_position(cone_idx)
                yaw = np.random.uniform(0, 2 * math.pi)
                origin = self.scene.env_origins[env_idx]
                root_pose[local_i, 0] = float(x) + float(origin[0])
                root_pose[local_i, 1] = float(y) + float(origin[1])
                root_pose[local_i, 2] = 0.1 + float(origin[2])
                root_pose[local_i, 3] = float(math.cos(float(yaw) / 2.0))
                root_pose[local_i, 4] = 0.0
                root_pose[local_i, 5] = 0.0
                root_pose[local_i, 6] = float(math.sin(float(yaw) / 2.0))
            cone_obj.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)
            cone_obj.write_root_velocity_to_sim(root_vel, env_ids=env_ids_t)
            cone_obj.reset(env_ids=env_ids_t)

    def _apply_goal_layout(self, env_ids: np.ndarray, sampled_layouts: dict) -> None:
        """Update goal positions and visual markers for the selected environments.

        Store env-local goal coordinates in the GPU tensor and update the
        corresponding USD goal markers.
        """
        stage = omni.usd.get_context().get_stage()
        for env_idx in env_ids:
            gx, gy = sampled_layouts[int(env_idx)]["goal"]
            self._goal_positions[env_idx, 0] = gx
            self._goal_positions[env_idx, 1] = gy
            path = f"/World/envs/env_{env_idx}/Venue/GoalMarker"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(float(gx), float(gy), 0.01))

    # --- Scene construction helpers ---

    def _prim_exists(self, prim_path: str) -> bool:
        return omni.usd.get_context().get_stage().GetPrimAtPath(prim_path).IsValid()

    def _spawn_wall(self, prim_path: str, start, end, height: float, thickness: float, color=(0.8, 0.8, 0.8)) -> None:
        if self._prim_exists(prim_path):
            return
        sx, sy, _ = start
        ex, ey, _ = end
        cx, cy = (sx + ex) / 2.0, (sy + ey) / 2.0
        cz = height / 2.0
        length = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
        yaw = math.atan2(ey - sy, ex - sx)
        cfg = sim_utils.CuboidCfg(
            size=(length, thickness, height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        cfg.func(
            prim_path, cfg, translation=(cx, cy, cz), orientation=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        )

    def _spawn_obstacles(self, prim_path: str) -> RigidObject:
        """Configure and instantiate a dynamic cone obstacle at the given prim path."""
        cone_cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.ConeCfg(
                radius=0.15,
                height=0.4,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        return RigidObject(cfg=cone_cfg)

    def _spawn_goal_markers(self) -> None:
        for env_idx in range(self.num_envs):
            path = f"/World/envs/env_{env_idx}/Venue/GoalMarker"
            if self._prim_exists(path):
                continue
            cfg = sim_utils.CylinderCfg(
                radius=0.2,
                height=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
            # Placeholder; _apply_goal_layout moves it on every reset.
            cfg.func(path, cfg, translation=(ARENA_CENTER[0], ARENA_CENTER[1], 0.01))

    def _spawn_venue(self) -> None:
        self.cone_objects: list[RigidObject] = []
        xn, xx = self.cfg.arena_x_min, self.cfg.arena_x_max
        yn, yx = self.cfg.arena_y_min, self.cfg.arena_y_max
        for env_idx in range(self.num_envs):
            parent = f"/World/envs/env_{env_idx}/Venue"
            if not self._prim_exists(parent):
                sim_utils.create_prim(parent, "Xform")
            # Named for the wall's position: North is at y_max, South at y_min.
            for name, start, end in (
                ("WallNorth", (xn, yx, 0), (xx, yx, 0)),
                ("WallWest", (xn, yx, 0), (xn, yn, 0)),
                ("WallSouth", (xn, yn, 0), (xx, yn, 0)),
                ("WallEast", (xx, yn, 0), (xx, yx, 0)),
            ):
                self._spawn_wall(f"{parent}/{name}", start, end, self.cfg.wall_height, self.cfg.wall_thickness)

        for cone_idx in range(self.cfg.num_obstacles):
            self.cone_objects.append(self._spawn_obstacles(f"/World/envs/env_.*/Venue/Cone{cone_idx}"))
