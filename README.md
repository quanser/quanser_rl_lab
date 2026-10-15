# Quanser RL Lab

Isaac Lab reinforcement-learning tasks for Quanser robots, trained with
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO.

> **Tested on** Isaac Sim 5.1.0, Isaac Lab 2.3.2, rsl-rl-lib 5.0.1, Python
> 3.11, Linux. Other combinations may work but are not exercised.

## Tasks

| Task | Robot | Environment ids |
|---|---|---|
| [Mapless navigation](docs/qbot_mapless_nav/README.md) | QBot Platform | `Quanser-QBot-Mapless-Nav-v0`, `Quanser-QBot-Mapless-Nav-Play-v0` |

## Install

Requires a working Isaac Lab installation and its Python environment. Robot
USDs are tracked in Git LFS, so fetch them before installing:

```bash
git lfs install && git lfs pull
python -m pip install -e source/quanser_rl_lab
```

**Keep the `-e`.** The robot USDs sit next to the Python package rather than
inside it, so a plain `pip install` copies the package and leaves them behind.

### Check the install

```bash
python scripts/list_envs.py
```

That should list the environments in the table above.

The test suite builds and steps every registered task:

```bash
python -m pytest tests/ -v
```

For an end-to-end check that the training works, run three iterations:

```bash
python scripts/rsl_rl/train.py --task Quanser-QBot-Mapless-Nav-v0 \
    --num_envs 64 --headless --max_iterations 3 --seed 1
```

## Scripts

```
scripts/
  rsl_rl/train.py, play.py        train and replay a checkpoint
  list_envs.py                    list registered environments
  zero_agent.py, random_agent.py  drive an env without a policy
tests/
  test_env_smoke.py               every registered task: builds, steps,
                                  obeys its declared observation space
  test_qbot_mapless_nav.py        checks specific to the QBot task
```

`test_env_smoke.py` discovers tasks from the gym registry, so a newly
registered robot is covered without editing it. Anything true of only one task
belongs in that task's own module.

## Repository layout

```
CHANGELOG.md                      release notes
docs/<task>/                      one page per task
source/quanser_rl_lab/
  data/robots/<ROBOT>/            robot USDs (Git LFS)
  quanser_rl_lab/
    actuators/                    stateful actuator models, stepped in the physics loop
    assets/                       ArticulationCfgs and USD path constants
    models/                       policy models and action distributions (loaded by name)
    tasks/direct/<task>/          env, env cfg and the rsl_rl PPO cfg
```

## Licence

Apache-2.0, except for files derived from the Isaac Lab template, which keep
their BSD-3-Clause headers. See [LICENSE](LICENSE),
[LICENSE-IsaacLab](LICENSE-IsaacLab) and
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
