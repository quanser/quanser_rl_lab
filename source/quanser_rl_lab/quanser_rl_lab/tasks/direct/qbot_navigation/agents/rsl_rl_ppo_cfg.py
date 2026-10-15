# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name: str = "qbot_mapless_nav"
    num_steps_per_env: int = 256
    max_iterations: int = 10_000
    save_interval: int = 100
    empirical_normalization: bool = False
    obs_groups: dict = {"actor": ["policy"], "critic": ["policy"]}

    # The older policy=RslRlPpoActorCriticCfg has no distribution_cfg,
    # so there is nowhere to attach the std cap below,
    # and std explodes over a few thousand iterations.
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="quanser_rl_lab.models.tanh_mean_mlp_model:TanhMeanMLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        # std_type="log" is load-bearing. rsl_rl's default "scalar" lets
        # std_param go negative and torch.normal then raises.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="quanser_rl_lab.models.capped_gaussian_distribution:CappedGaussianDistribution",
            init_std=1.0,
            std_type="log",
        ),
    )
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=10,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=0.5,
    )
