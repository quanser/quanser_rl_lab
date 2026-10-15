# Third-Party Licenses

This project is licensed under Apache-2.0 (see [LICENSE](LICENSE)). It also
contains code derived from third-party open-source projects, listed below.

## isaac-sim/IsaacLab

- Source: https://github.com/isaac-sim/IsaacLab
- License: BSD-3-Clause — full text in [LICENSE-IsaacLab](LICENSE-IsaacLab)
- Used in: the project scaffolding generated from the Isaac Lab extension
  template, and the runner scripts under `scripts/`

Files still carrying the Isaac Lab copyright header are those used
substantially unchanged from that template:

- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/play.py`
- `scripts/rsl_rl/cli_args.py`
- `scripts/list_envs.py`
- `scripts/zero_agent.py`
- `scripts/random_agent.py`
- `source/quanser_rl_lab/setup.py`
- the package and task `__init__.py` scaffolding

BSD-3-Clause requires that notice be retained, so it stays on those files.
Everything else in this repository is original work under Apache-2.0.

## leggedrobotics/rsl_rl

- Source: https://github.com/leggedrobotics/rsl_rl
- License: BSD-3-Clause
- Used as: a runtime dependency (PPO implementation)

Not vendored. `quanser_rl_lab/models/tanh_mean_mlp_model.py` and
`quanser_rl_lab/models/capped_gaussian_distribution.py` subclass `rsl_rl` types but
are original work.
