from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import fourier_gr3_flat_tracking_env_cfg
from .rl_cfg import fourier_gr3_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Fourier-GR3",
  env_cfg=fourier_gr3_flat_tracking_env_cfg(),
  play_env_cfg=fourier_gr3_flat_tracking_env_cfg(play=True),
  rl_cfg=fourier_gr3_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Fourier-GR3-No-State-Estimation",
  env_cfg=fourier_gr3_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=fourier_gr3_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=fourier_gr3_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
