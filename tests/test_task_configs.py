"""Generic tests for task config integrity."""

import pytest

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


@pytest.fixture(scope="module")
def all_task_ids() -> list[str]:
  """Get all registered task IDs."""
  return list_tasks()


def test_all_tasks_loadable(all_task_ids: list[str]) -> None:
  """All registered tasks should be loadable without errors."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}': {e}")


def test_all_tasks_have_play_config(all_task_ids: list[str]) -> None:
  """All tasks should be loadable in play mode."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id, play=True)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} play mode did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}' in play mode: {e}")


def test_play_mode_episode_length(all_task_ids: list[str]) -> None:
  """Play mode tasks should have infinite episode length."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.episode_length_s >= 1e9, (
      f"{task_id} (play mode) episode_length_s={cfg.episode_length_s}, expected >= 1e9"
    )


def test_play_mode_observation_corruption_disabled(all_task_ids: list[str]) -> None:
  """Play mode tasks should have observation corruption disabled for policy."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)

    assert "actor" in cfg.observations, (
      f"Play mode task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Play mode task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert not policy_obs.enable_corruption, (
      f"Play mode task {task_id} has enable_corruption=True, expected False"
    )


def test_training_mode_observation_corruption_enabled(all_task_ids: list[str]) -> None:
  """Training mode tasks should have observation corruption enabled for policy."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    assert "actor" in cfg.observations, (
      f"Training task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Training task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert policy_obs.enable_corruption, (
      f"Training task {task_id} has enable_corruption=False, expected True"
    )


def test_critic_observation_corruption_always_disabled(all_task_ids: list[str]) -> None:
  """Critic observations should always have corruption disabled."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    if "critic" not in cfg.observations:
      continue

    critic_obs = cfg.observations["critic"]
    assert isinstance(critic_obs, ObservationGroupCfg), (
      f"Task {task_id} critic observation is not ObservationGroupCfg"
    )

    assert not critic_obs.enable_corruption, (
      f"Task {task_id} has critic enable_corruption=True, expected False"
    )


def test_play_training_observation_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching observation structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    # Same observation groups.
    assert set(training_cfg.observations.keys()) == set(play_cfg.observations.keys()), (
      f"Observation groups mismatch between {task_id} training and play modes"
    )

    # Same observation terms within each group.
    for obs_group_name in training_cfg.observations:
      training_terms = set(training_cfg.observations[obs_group_name].terms.keys())
      play_terms = set(play_cfg.observations[obs_group_name].terms.keys())

      assert training_terms == play_terms, (
        f"Observation terms mismatch in group '{obs_group_name}' "
        f"between {task_id} training and play modes"
      )


def test_play_training_action_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching action structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    assert set(training_cfg.actions.keys()) == set(play_cfg.actions.keys()), (
      f"Action structure mismatch between {task_id} training and play modes"
    )


def test_play_mode_disables_push_robot(all_task_ids: list[str]) -> None:
  """Play mode tasks should disable push_robot event."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert "push_robot" not in cfg.events, (
      f"Play mode task {task_id} has push_robot event, expected it to be removed"
    )


def test_step_boundary_rewards_only_on_target_heading_teacher(
  all_task_ids: list[str],
) -> None:
  """Privileged stair geometry rewards should stay scoped to the perceptive teacher."""
  step_reward_names = {
    "foot_landing_flatness_penalty",
    "foot_step_lip_volume_penalty",
    "heel_step_riser_clearance_penalty",
    "shank_step_lip_proximity_penalty",
    "toe_step_riser_slab_penalty",
  }
  target_task = "Mjlab-Velocity-TargetHeading-Rough-Teacher-Unitree-G1"
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)
    present = step_reward_names.intersection(cfg.rewards)
    if task_id == target_task:
      assert present == step_reward_names
      assert cfg.rewards["foot_step_lip_volume_penalty"].weight == -3.2
      assert cfg.rewards["foot_step_lip_volume_penalty"].params[
        "edge_radius"
      ] == 0.07
      assert cfg.rewards["foot_step_lip_volume_penalty"].params[
        "min_terrain_level"
      ] == 3
      assert cfg.rewards["foot_step_lip_volume_penalty"].params[
        "nearest_boundaries"
      ] == 4
      assert cfg.rewards["foot_step_lip_volume_penalty"].params[
        "support_speed_floor"
      ] == 0.08
      assert cfg.rewards["toe_step_riser_slab_penalty"].params[
        "min_terrain_level"
      ] == 3
      assert cfg.rewards["toe_step_riser_slab_penalty"].params[
        "nearest_boundaries"
      ] == 4
      assert cfg.rewards["toe_step_riser_slab_penalty"].params[
        "slab_depth"
      ] == 0.10
      assert cfg.rewards["toe_step_riser_slab_penalty"].params[
        "toe_x_min"
      ] == 0.08
      assert cfg.rewards["toe_step_riser_slab_penalty"].params[
        "approach_speed_floor"
      ] == 0.08
      assert cfg.rewards["toe_step_riser_slab_penalty"].weight == -4.2
      assert cfg.rewards["heel_step_riser_clearance_penalty"].params[
        "min_terrain_level"
      ] == 3
      assert cfg.rewards["heel_step_riser_clearance_penalty"].params[
        "nearest_boundaries"
      ] == 4
      assert cfg.rewards["heel_step_riser_clearance_penalty"].params[
        "heel_clearance"
      ] == 0.10
      assert cfg.rewards["heel_step_riser_clearance_penalty"].params[
        "heel_x_max"
      ] == 0.0
      assert cfg.rewards["heel_step_riser_clearance_penalty"].weight == -3.5
      assert cfg.rewards["foot_landing_flatness_penalty"].params[
        "min_terrain_level"
      ] == 3
      assert cfg.rewards["foot_landing_flatness_penalty"].params[
        "near_height"
      ] == 0.15
      assert cfg.rewards["foot_landing_flatness_penalty"].params[
        "max_tilt_deg"
      ] == 12.0
      assert cfg.rewards["foot_landing_flatness_penalty"].params[
        "max_upward_speed"
      ] == 0.10
      assert cfg.rewards["foot_landing_flatness_penalty"].params[
        "height_sensor_name"
      ] == "foot_height_scan"
      assert cfg.rewards["foot_landing_flatness_penalty"].params[
        "contact_sensor_name"
      ] == "feet_ground_contact"
      assert cfg.rewards["foot_landing_flatness_penalty"].weight == -2.0
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "min_terrain_level"
      ] == 3
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "nearest_boundaries"
      ] == 4
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "clearance_radius"
      ] == 0.20
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "collision_radius"
      ] == 0.05
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "collision_weight"
      ] == 4.0
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "height_history_len"
      ] == 6
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "height_gain_threshold"
      ] == 0.03
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "ascent_hold_steps"
      ] == 4
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "shank_tilt_threshold_deg"
      ] == 15.0
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "shank_grid_shape"
      ] == (1, 3, 5)
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "shank_x_range"
      ] == (0.045, 0.045)
      assert cfg.rewards["shank_step_lip_proximity_penalty"].params[
        "shank_z_range"
      ] == (-0.23, -0.10)
      assert cfg.rewards["shank_step_lip_proximity_penalty"].weight == -1.2
    else:
      assert not present, f"{task_id} unexpectedly enables {sorted(present)}"
