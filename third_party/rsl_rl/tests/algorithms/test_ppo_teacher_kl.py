"""Tests for PPO teacher-guidance losses."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_teacher_kl import PPOTeacherKL
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 2
OBS_DIM = 8
NUM_ACTIONS = 3


def _build_teacher_kl(loss_cfg: dict) -> PPOTeacherKL:
    obs = TensorDict(
        {
            "actor": torch.zeros(NUM_ENVS, OBS_DIM),
            "critic": torch.zeros(NUM_ENVS, OBS_DIM),
            "teacher": torch.zeros(NUM_ENVS, OBS_DIM),
        },
        batch_size=[NUM_ENVS],
    )
    obs_groups = {
        "actor": ["actor"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[16])
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    return PPOTeacherKL(actor, critic, storage, teacher_kl_cfg=loss_cfg)


def test_mean_huber_guidance_ignores_std_mismatch() -> None:
    """Mean-only guidance should not penalize different teacher/student std."""
    alg = _build_teacher_kl({"loss_type": "mean_huber", "huber_delta": 0.5})
    mean = torch.zeros(2, NUM_ACTIONS, requires_grad=True)
    teacher_params = (torch.zeros(2, NUM_ACTIONS), torch.full((2, NUM_ACTIONS), 2.0))
    student_params = (mean, torch.full((2, NUM_ACTIONS), 0.25))

    loss, logs = alg._compute_mean_teacher_loss(teacher_params, student_params)

    assert loss.item() == 0.0
    assert logs["teacher_mean_huber"].item() == 0.0
    assert logs["teacher_kl"].item() > 0.0


def test_mean_huber_guidance_applies_loss_cap() -> None:
    """The update loss should respect max_teacher_loss for mean guidance."""
    alg = _build_teacher_kl(
        {
            "loss_type": "mean_huber",
            "huber_delta": 0.5,
            "max_teacher_loss": 0.25,
        }
    )
    teacher_params = (torch.zeros(1, NUM_ACTIONS), torch.ones(1, NUM_ACTIONS))
    student_params = (
        torch.full((1, NUM_ACTIONS), 10.0, requires_grad=True),
        torch.ones(1, NUM_ACTIONS),
    )

    loss, logs = alg._compute_mean_teacher_loss(teacher_params, student_params)

    assert loss.item() == 0.25
    assert logs["teacher_loss_for_update"].item() == 0.25
    assert logs["teacher_mean_huber"].item() > loss.item()
