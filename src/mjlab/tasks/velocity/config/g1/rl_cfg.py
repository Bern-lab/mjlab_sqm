"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlPpoTeacherKLAlgorithmCfg,
  RslRlTeacherKLCfg,
  RslRlTeacherKLRunnerCfg,
)

G1_TEACHER_KL_CHECKPOINT = (
  "logs/rsl_rl/g1_velocity_target_heading_teacher_depth/"
  "Mjlab-Velocity-TargetHeading-Rough-Teacher-Unitree-G1/"
  "2026-05-21_11-54-52_rollback_29750/model_118200.pt"
)

_DEPTH_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_DEPTH_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


def _unitree_g1_policy_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def _unitree_g1_depth_policy_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=_DEPTH_CNN_CFG,
    class_name=_DEPTH_MODEL_CLS,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=_unitree_g1_policy_model_cfg(),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=40_001,
  )


def unitree_g1_target_heading_teacher_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO config for the target-heading teacher policy task."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor = _unitree_g1_depth_policy_model_cfg()
  cfg.obs_groups = {
    "actor": ("actor", "camera"),
    "critic": ("critic",),
  }
  cfg.experiment_name = "g1_velocity_target_heading_teacher_depth"
  return cfg


def unitree_g1_blind_rough_teacherkl_runner_cfg() -> RslRlTeacherKLRunnerCfg:
  """Create PPO + frozen-teacher-KL config for Unitree G1 blind rough training."""
  return RslRlTeacherKLRunnerCfg(
    actor=_unitree_g1_policy_model_cfg(),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    teacher=_unitree_g1_depth_policy_model_cfg(),
    algorithm=RslRlPpoTeacherKLAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      teacher_kl_cfg=RslRlTeacherKLCfg(
        checkpoint_path=G1_TEACHER_KL_CHECKPOINT,
        loss_type="mean_huber",
        lambda_start=0.05,
        lambda_end=0.0,
        warmup_iters=1000,
        constant_iters=0,
        anneal_iters=8000,
        schedule="cosine",
        huber_delta=0.5,
        max_teacher_loss=3.0,
        max_kl_loss=None,
        max_kl_loss_tail_slope=0.0,
        check_shapes=True,
        fail_on_nonfinite_kl=True,
        debug_shapes=False,
      ),
    ),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher", "camera"),
    },
    experiment_name="g1_blind_rough_teacherkl",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=40_001,
  )


def unitree_g1_blind_rough_target_navigation_teacherkl_runner_cfg() -> (
  RslRlTeacherKLRunnerCfg
):
  """Create Teacher-KL config for blind rough target-navigation training."""
  cfg = unitree_g1_blind_rough_teacherkl_runner_cfg()
  cfg.experiment_name = "g1_blind_rough_target_navigation_teacherkl"
  return cfg
