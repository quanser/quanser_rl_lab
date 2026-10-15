# QBot Platform — mapless navigation

`Quanser-QBot-Mapless-Nav-v0` · `Quanser-QBot-Mapless-Nav-Play-v0`

A differential-drive robot reaches a randomly placed goal in a walled arena
while avoiding cone obstacles, using only a 2D lidar scan, wheel odometry, a
gyro and a noisy goal pose. No map.

| | |
|---|---|
| Arena | 4.0 × 4.0 m, walled, 9 cone obstacles (radius 0.15 m) |
| Episode | 30 s (300 control steps), ends on goal, collision or timeout |
| Success | robot centre within 0.25 m of the goal, at any speed |
| Failure | any contact on the chassis above 0.2 N |
| Control rate | 10 Hz (`decimation=10` against a 100 Hz sim) |

Robot, goal and obstacle positions are resampled every episode.

## Run it

```bash
# train
python scripts/rsl_rl/train.py --task Quanser-QBot-Mapless-Nav-v0 --num_envs 256 --headless

# evaluate a checkpoint at full 9-cone density, and export it
python scripts/rsl_rl/play.py --task Quanser-QBot-Mapless-Nav-Play-v0 --load_run <run-dir>
```

Any config field can be overridden on the command line, e.g.
`env.num_obstacles=4` or `env.arena_width=6.0`.

Use the `-Play-v0` id for both evaluation and export. The obstacle curriculum
advances only during training and its level is not stored in the checkpoint,
so the training id would replay a 9-cone policy at the 2-cone starting level.

Deployment consumes two things from the run directory:

```
<run-dir>/
  exported/policy.pt, policy.onnx   written by play.py
  params/env.yaml                   written by train.py at the start of the run
```

`env.yaml` carries the keys deployment needs alongside the weights:
`max_lin_vel` and `max_ang_vel` to scale the policy output, and `decimation`
to derive the control rate.

## Observation space — 47 values, all normalised

| Slice | Content | Normalisation |
|---|---|---|
| `0:42` | lidar ranges, 42 rays over 360°, 8.6° apart | ÷ 8.0 m |
| `42` | goal distance | ÷ 8.0 m |
| `43` | goal heading | ÷ π |
| `44:46` | left, right wheel speed | ÷ 15.75 rad/s |
| `46` | gyro yaw rate | ÷ 2.40 rad/s |

Ray `k` points at `-180° + k × 8.571°`, increasing counter-clockwise, so ray 21 is straight ahead.

Sensor noise is modelled: lidar range σ = 0.02 m with 2% dropout, wheel,
gyro and localisation noise fitted from hardware data.
Observation latency is **not** modelled, training sees perfectly fresh data.

## Action space — 2 values

| Index | Meaning | Range |
|---|---|---|
| `0` | forward velocity | ± 0.6 m/s |
| `1` | yaw rate | ± 2.4 rad/s |

The raw policy output is bounded between (-1, 1); the env scales by the limits above and
clamps. Deployment must apply the same scale, read from `max_lin_vel` and
`max_ang_vel` in the saved params (e.g. `env.yaml`). Downstream of that, commands pass through
differential-drive inverse kinematics, a wheel deadband, asymmetric
acceleration limits and a stiction model, all fitted to the physical robot behaviour.

## Rewards

| Term | Value | When |
|---|---|---|
| Goal reached | **+150** | once, terminal |
| Collision | **−250** | once, terminal |
| Timeout | **−100** | once, terminal |
| Progress | **+300 / m** | per step, potential-based on distance closed |
| Time | **−0.5** | every step |
| Obstacle proximity | **−1.0 × max(0, 1.35 − min scan)** | every step |

Time and obstacle proximity are per-control-step and must be rescaled by hand
if `decimation` changes; the rest are per-event or per-metre and must not.

The collision penalty is sized so crashing is never cheaper than timing out: a
timeout costs −100 plus 300 steps of time penalty (−150), about −250 either way.

## Curriculum

Starts at 2 cones and adds one each time the success rate over the last 500
episodes reaches 0.70, up to 9. Cones not in play are parked outside the arena
walls, where the walls occlude them from the lidar.

## Results

Three seeds at `num_envs=256`, ~9000 iterations, mean over the last 50:

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| success | 0.842 | 0.812 | 0.848 |
| collision | 0.158 | 0.188 | 0.152 |
| timeout | 0.000 | 0.000 | 0.000 |

![Success rate against training iteration for three seeds, rising from about
0.35 to a plateau near 0.84](success_rate.png)

Training runs at about 220 iterations per hour per GPU at `num_envs=256`.

## Known limitations

- **No observation latency modelled.** Real lidar staleness is ~61 ms at
  10 Hz; simulation is optimistic here, so hardware may underperform the
  success rates quoted above.
