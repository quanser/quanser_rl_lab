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

## ROBOTIS-GIT/cyclo_lab

- Source: https://github.com/ROBOTIS-GIT/cyclo_lab (taken from it under its
  former name, `robotis_lab`, which now redirects there)
- License: Apache-2.0
- Used in: the QArm Research reach task, ported from that project's OMY
  reach task, and the QArm Research robot description, derived from its
  `OMY.usd`

Files derived from it keep the ROBOTIS copyright header alongside a note of
Quanser's modifications, as Apache-2.0 section 4 requires:

- `source/quanser_rl_lab/quanser_rl_lab/assets/qarm_research.py`
- `source/quanser_rl_lab/quanser_rl_lab/tasks/manager_based/qarm_research_reach/__init__.py`
- `source/quanser_rl_lab/quanser_rl_lab/tasks/manager_based/qarm_research_reach/reach_env_cfg.py`
- `source/quanser_rl_lab/quanser_rl_lab/tasks/manager_based/qarm_research_reach/joint_pos_env_cfg.py`
- `source/quanser_rl_lab/quanser_rl_lab/tasks/manager_based/qarm_research_reach/mdp/__init__.py`
- `source/quanser_rl_lab/quanser_rl_lab/tasks/manager_based/qarm_research_reach/mdp/rewards.py`
- `source/quanser_rl_lab/quanser_rl_lab/tasks/manager_based/qarm_research_reach/agents/rsl_rl_ppo_cfg.py`

`source/quanser_rl_lab/data/robots/QArmResearch/qarm_research.usd` is a
Quanser-created derivative of that project's `data/robots/OMY/OMY.usd`. The
QArm Research is physically a ROBOTIS OMY, so all visual and collision mesh
geometry is ROBOTIS's, unmodified -- no vertex differs.

Changed by Quanser, as Apache-2.0 section 4(b) requires:

- **The link frames were redefined in DH convention.** All six arm joints
  differ from `OMY.usd` in `physics:axis`, `physics:localPos0` or
  `physics:localRot0`, and all six links differ in `physics:centerOfMass` and
  transform, with `link2`, `link3`, `link4` and `link6` also carrying reordered
  `physics:diagonalInertia` and `physics:principalAxes`.


## leggedrobotics/rsl_rl

- Source: https://github.com/leggedrobotics/rsl_rl
- License: BSD-3-Clause
- Used as: a runtime dependency (PPO implementation)

Not vendored. `quanser_rl_lab/models/tanh_mean_mlp_model.py` and
`quanser_rl_lab/models/capped_gaussian_distribution.py` subclass `rsl_rl` types but
are original work.
