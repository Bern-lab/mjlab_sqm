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
G1_TARGET_HEADING_DEPTH_TEACHER_KL_CHECKPOINT = G1_TEACHER_KL_CHECKPOINT
G1_LSTM_TEACHER_KL_NUM_STEPS_PER_ENV = 24
"""Rollout horizon per env for each PPO update.

G1 velocity env step is 0.02s, so 24 steps cover about 0.48s per rollout.
"""

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
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=100_001,
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


def _unitree_g1_critic_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
  )


def _unitree_g1_lstm_policy_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    class_name="RNNModel",
    rnn_type="lstm",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def _unitree_g1_teacher_guidance_cfg() -> RslRlTeacherKLCfg:#同时影响此分支所有student
  """Match main's normal TeacherKL setting: PPO and teacher guidance both active."""
  return RslRlTeacherKLCfg(
    enabled=True,
    imitation_only=False,
    imitation_loss_coef=1.0,
    checkpoint_path=G1_TEACHER_KL_CHECKPOINT,
    loss_type="mean_huber",
    lambda_start=0.03,
    lambda_end=0.0,
    warmup_iters=1000,
    constant_iters=0,
    anneal_iters=10000,
    schedule="cosine",
    huber_delta=0.5,
    max_teacher_loss=None,
    max_kl_loss=None,
    max_kl_loss_tail_slope=0.0,
    check_shapes=True,
    fail_on_nonfinite_kl=True,
    debug_shapes=False,
  )


def _unitree_g1_teacher_student_runner_cfg(
  *,
  actor: RslRlModelCfg,
  experiment_name: str,
  num_steps_per_env: int = 24,
) -> RslRlTeacherKLRunnerCfg:
  """Create the shared main-style PPO + frozen-teacher guidance runner."""
  return RslRlTeacherKLRunnerCfg(
    actor=actor,
    critic=_unitree_g1_critic_model_cfg(),
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
      teacher_kl_cfg=_unitree_g1_teacher_guidance_cfg(),
    ),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher", "camera"),
    },
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=num_steps_per_env,
    max_iterations=40_001,
  )


def unitree_g1_blind_rough_teacherkl_runner_cfg() -> RslRlTeacherKLRunnerCfg:
  """Create main-style PPO + frozen-teacher guidance config for blind rough."""
  return _unitree_g1_teacher_student_runner_cfg(
    actor=_unitree_g1_policy_model_cfg(),
    experiment_name="g1_blind_rough_teacherkl",
  )


def unitree_g1_blind_stairs_flag_teacherkl_runner_cfg() -> RslRlTeacherKLRunnerCfg:
  """Create main-style PPO + frozen-teacher guidance config for stair-flag."""
  return _unitree_g1_teacher_student_runner_cfg(
    actor=_unitree_g1_policy_model_cfg(),
    experiment_name="g1_blind_stairs_flag_teacherkl",
  )


def unitree_g1_blind_stairs_flag_lstm_teacherkl_runner_cfg(
  num_steps_per_env: int = G1_LSTM_TEACHER_KL_NUM_STEPS_PER_ENV,
) -> RslRlTeacherKLRunnerCfg:
  """Create main-style LSTM PPO + frozen-teacher guidance config for stair-flag."""
  return _unitree_g1_teacher_student_runner_cfg(
    actor=_unitree_g1_lstm_policy_model_cfg(),
    experiment_name="g1_blind_stairs_flag_lstm_teacherkl",
    num_steps_per_env=num_steps_per_env,
  )


def unitree_g1_blind_rough_lstm_teacherkl_runner_cfg(
  num_steps_per_env: int = G1_LSTM_TEACHER_KL_NUM_STEPS_PER_ENV,
) -> RslRlTeacherKLRunnerCfg:
  """Create main-style LSTM PPO + frozen-teacher guidance config for blind rough."""
  return _unitree_g1_teacher_student_runner_cfg(
    actor=_unitree_g1_lstm_policy_model_cfg(),
    experiment_name="g1_blind_rough_lstm_teacherkl",
    num_steps_per_env=num_steps_per_env,
  )
