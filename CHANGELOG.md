# Changelog

## 0.1.0 (2026-10-15)

Initial release.

### Added

- `Quanser-QBot-Mapless-Nav-v0` — QBot Platform mapless navigation: reach a
  randomly placed goal in a walled 4×4 m arena with up to nine cone obstacles,
  from a 42-ray 2D lidar scan, wheel odometry, a gyro and a noisy goal
  bearing. Policy and hardware run at 10 Hz.
- `Quanser-QBot-Mapless-Nav-Play-v0` — the same task with the obstacle
  curriculum disabled, so evaluation runs at full density. Use this id with
  `play.py`; it shares the training run's log directory.
- Obstacle-count curriculum gated on success rate, ramping 2 → 9 cones.
- Sensor and actuator models fitted to the physical platform: wheel deadband,
  asymmetric acceleration limits, a stiction filter, and lidar, wheel, gyro
  and localisation noise.
- `scripts/` — `rsl_rl/train.py`, `rsl_rl/play.py`, `list_envs.py`,
  `zero_agent.py` and `random_agent.py`.
- `tests/` — a smoke test covering every registered task, plus checks specific
  to the QBot task.

### Known limitations

- **Install with `pip install -e`.** Robot USDs live in
  `source/quanser_rl_lab/data/`, next to the Python package rather than inside
  it, so a plain `pip install` copies the package and leaves the robots
  behind. The USDs are tracked in Git LFS and need `git lfs pull` before first
  use.
- No observation latency modelled. Real lidar staleness is ~61 ms at 10 Hz, so
  simulation is optimistic here and hardware may underperform the success
  rates in [docs/qbot_mapless_nav/README.md](docs/qbot_mapless_nav/README.md).

### Tested with

Isaac Sim 5.1.0, Isaac Lab 2.3.2, rsl-rl-lib 5.0.1, Python 3.11, Linux.
