from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .blind_rough_lstm_teacher_kl_env_cfg import (
  unitree_g1_blind_rough_lstm_teacherkl_env_cfg,
)
from .blind_rough_teacher_kl_env_cfg import unitree_g1_blind_rough_teacherkl_env_cfg
from .env_cfgs import (
  unitree_g1_blind_rough_env_cfg,
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .env_cfgs import (
  unitree_g1_blind_target_heading_rough_env_cfg as unitree_g1_blind_target_heading_rough_env_cfg,
)
from .rl_cfg import (
  unitree_g1_blind_rough_lstm_teacherkl_runner_cfg,
  unitree_g1_blind_rough_teacherkl_runner_cfg,
  unitree_g1_ppo_runner_cfg,
  unitree_g1_target_heading_teacher_runner_cfg,
)
from .target_heading_teacher_env_cfg import unitree_g1_target_heading_teacher_env_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_teacherkl_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_teacherkl_env_cfg(play=True),
  rl_cfg=unitree_g1_blind_rough_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_lstm_teacherkl_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_lstm_teacherkl_env_cfg(play=True),
  rl_cfg=unitree_g1_blind_rough_lstm_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)



register_mjlab_task(
  task_id="Mjlab-Velocity-TargetHeading-Rough-Teacher-Unitree-G1",
  env_cfg=unitree_g1_target_heading_teacher_env_cfg(),
  play_env_cfg=unitree_g1_target_heading_teacher_env_cfg(play=True),
  rl_cfg=unitree_g1_target_heading_teacher_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
