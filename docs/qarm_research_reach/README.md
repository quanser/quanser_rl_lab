# QArm Research — end-effector reach

`Quanser-QArmResearch-Reach-v0` · `Quanser-QArmResearch-Reach-Play-v0`

A 6-DOF arm drives its end effector to a commanded pose — position and
orientation — sampled from one of two disjoint zones on either side of the
robot. Joint position control, no gripper, no obstacles beyond a fixed table.

This is a starting point to build on rather than a tuned task: the reward
terms, goal zones and PPO settings below are a working baseline, and are meant
to be changed.

| | |
|---|---|
| Robot | QArm Research, 6 revolute joints, gripper held at its default pose |
| Scene | Ground plane and a fixed table at (0.6, 0, 0) in the robot root frame |
| Episode | 16 s (480 control steps), ends on timeout only |
| Goal | Resampled every 3–5 s, so roughly 4 goals per episode |
| Control rate | 30 Hz (`decimation=2` against a 60 Hz sim) |
| Tracked frame | `link6`, the wrist — not the gripper tip |
| Success | Not defined — the reward is continuous tracking error |

## Run it

```bash
# train
python scripts/rsl_rl/train.py --task Quanser-QArmResearch-Reach-v0 --num_envs 1024 --headless

# evaluate a checkpoint
python scripts/rsl_rl/play.py --task Quanser-QArmResearch-Reach-Play-v0 --load_run <run-dir>
```

`-Play-v0` reduces the scene to 50 environments and turns off observation
noise. It samples goals from the same two zones as training.

A trained policy is attached to the [latest release][rel] as `policy.zip`.
Extract it at the root of the repository so its run directory lands under
`logs/rsl_rl/qarm_research_reach/`, then pass that directory's name to
`--load_run`. With no `--checkpoint`, `play.py` loads the newest checkpoint in
it.

[rel]: https://github.com/quanser/quanser_rl_lab/releases

## Observation space — 25 values

| Values | Term | Notes |
|---|---|---|
| 6 | `joint_pos_rel` | arm joints, relative to the default pose, ±0.01 uniform noise |
| 6 | `joint_vel_rel` | arm joints, relative to default, ±0.01 uniform noise |
| 7 | `pose_command` | goal position (3) and quaternion (4), in the robot root frame |
| 6 | `last_action` | previous action vector |

Nothing is normalised, and the end-effector pose is not observed — the policy
infers it from joint angles. Noise is disabled in `-Play-v0`.

## Action space — 6 values

Joint position targets for `joint1`–`joint6`, as an offset from the default
pose: `target = default + 0.5 * action`. Actions are **not** clipped, so the
scale is a gain rather than a bound. The scaled action will ultimately be
limited by the joint limits. The gripper joints are actuated but not commanded; 
they hold their default position.

## Rewards

| Term | Weight | Notes |
|---|---|---|
| `end_effector_position_tracking` | -0.25 | L2 distance from `link6` to the goal |
| `end_effector_position_tracking_fine_grained` | +0.12 | `1 - tanh(d / 0.1)`, so it only pays inside ~10 cm |
| `end_effector_orientation_tracking` | -0.12 | shortest-path quaternion error, in radians |
| `arm_contact` | -0.25 | per arm link over 1.0 N, covering self-collision and the table |
| `action_rate` | -0.0001 → -0.005 | raised by the curriculum |
| `joint_vel` | -0.0001 → -0.001 | raised by the curriculum |

**The tracked body is `link6`, the wrist, not the gripper tip.** The gripper
extends roughly 0.19 m beyond it, so a policy trained here places the wrist on the
commanded pose and the fingers reach past it by about that much. 

The USD does carry `end_effector_flange_link` and `end_effector_link` prims 
under `link6`, but neither has a rigid-body API, so Isaac Lab does not expose 
them as bodies and they cannot be tracked as they stand. Targeting the tool 
point instead would mean giving one of them a rigid body, or adding a fixed 
offset frame, and re-tuning the goal zones around the new reach.


## Curriculum

`action_rate` and `joint_vel` are raised once `common_step_counter` passes 4500.