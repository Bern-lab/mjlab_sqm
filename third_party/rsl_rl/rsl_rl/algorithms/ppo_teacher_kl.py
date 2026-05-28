# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class PPOTeacherKL(PPO):
    """PPO with an additional frozen-teacher guidance term."""

    teacher: MLPModel | None
    """The frozen teacher actor model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        teacher_kl_cfg: dict | None = None,
        teacher_checkpoint_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO and store teacher-guidance configuration."""
        super().__init__(actor, critic, storage, **kwargs)

        self.teacher = None
        self.teacher_loaded = False
        self.teacher_kl_cfg = dict(teacher_kl_cfg or {})
        self._teacher_kl_shapes_checked = False
        self.teacher_guidance_enabled = bool(self.teacher_kl_cfg.get("enabled", True))
        self.teacher_imitation_only = bool(self.teacher_kl_cfg.get("imitation_only", False))
        if self.teacher_imitation_only and not self.teacher_guidance_enabled:
            raise ValueError("teacher_kl_cfg.imitation_only requires enabled=True.")
        self.teacher_imitation_loss_coef = float(self.teacher_kl_cfg.get("imitation_loss_coef", 1.0))
        if self.teacher_imitation_loss_coef <= 0.0:
            raise ValueError("teacher_kl_cfg.imitation_loss_coef must be positive.")

        # Allow checkpoint path to be supplied either inside teacher_kl_cfg or as a direct keyword.
        self.teacher_checkpoint_path: str | None = None
        self.set_teacher_checkpoint(teacher_checkpoint_path or self.teacher_kl_cfg.get("checkpoint_path"))

        self.teacher_kl_lambda_start = float(self.teacher_kl_cfg.get("lambda_start", 1.0))
        self.teacher_kl_lambda_end = float(self.teacher_kl_cfg.get("lambda_end", 0.0))
        self.teacher_kl_warmup_iters = int(self.teacher_kl_cfg.get("warmup_iters", 0))
        self.teacher_kl_constant_iters = int(self.teacher_kl_cfg.get("constant_iters", 0))
        self.teacher_kl_anneal_iters = int(self.teacher_kl_cfg.get("anneal_iters", 3000))
        self.teacher_kl_schedule = str(self.teacher_kl_cfg.get("schedule", "linear"))
        self.teacher_kl_iteration = int(self.teacher_kl_cfg.get("iteration", 0))
        self.teacher_guidance_loss_type = str(self.teacher_kl_cfg.get("loss_type", "kl"))
        if self.teacher_guidance_loss_type not in {"kl", "mean_mse", "mean_huber"}:
            raise ValueError(f"Unsupported teacher guidance loss_type: {self.teacher_guidance_loss_type}")
        self.teacher_guidance_huber_delta = float(self.teacher_kl_cfg.get("huber_delta", 1.0))
        if self.teacher_guidance_huber_delta <= 0.0:
            raise ValueError("teacher_kl_cfg.huber_delta must be positive.")
        max_teacher_loss = self.teacher_kl_cfg.get("max_teacher_loss", self.teacher_kl_cfg.get("max_loss"))
        self.teacher_guidance_max_loss = None if max_teacher_loss is None else float(max_teacher_loss)

        self.teacher_kl_cfg.update(
            {
                "enabled": self.teacher_guidance_enabled,
                "imitation_only": self.teacher_imitation_only,
                "imitation_loss_coef": self.teacher_imitation_loss_coef,
                "loss_type": self.teacher_guidance_loss_type,
                "lambda_start": self.teacher_kl_lambda_start,
                "lambda_end": self.teacher_kl_lambda_end,
                "warmup_iters": self.teacher_kl_warmup_iters,
                "constant_iters": self.teacher_kl_constant_iters,
                "anneal_iters": self.teacher_kl_anneal_iters,
                "schedule": self.teacher_kl_schedule,
                "huber_delta": self.teacher_guidance_huber_delta,
                "max_teacher_loss": self.teacher_guidance_max_loss,
                "log_kl_when_lambda_zero": self.teacher_kl_cfg.get("log_kl_when_lambda_zero", True),
            }
        )

    def set_teacher_checkpoint(self, checkpoint_path: str | None) -> None:
        """Set or clear the checkpoint path used to initialize the frozen teacher."""
        self.teacher_checkpoint_path = checkpoint_path
        if checkpoint_path is None:
            self.teacher_kl_cfg.pop("checkpoint_path", None)
        else:
            self.teacher_kl_cfg["checkpoint_path"] = checkpoint_path

    def set_teacher_kl_schedule(
        self,
        lambda_start: float | None = None,
        lambda_end: float | None = None,
        warmup_iters: int | None = None,
        constant_iters: int | None = None,
        anneal_iters: int | None = None,
        schedule: str | None = None,
    ) -> None:
        """Update the teacher-KL weight schedule."""
        if lambda_start is not None:
            self.teacher_kl_lambda_start = float(lambda_start)
            self.teacher_kl_cfg["lambda_start"] = self.teacher_kl_lambda_start
        if lambda_end is not None:
            self.teacher_kl_lambda_end = float(lambda_end)
            self.teacher_kl_cfg["lambda_end"] = self.teacher_kl_lambda_end
        if warmup_iters is not None:
            self.teacher_kl_warmup_iters = int(warmup_iters)
            self.teacher_kl_cfg["warmup_iters"] = self.teacher_kl_warmup_iters
        if constant_iters is not None:
            self.teacher_kl_constant_iters = int(constant_iters)
            self.teacher_kl_cfg["constant_iters"] = self.teacher_kl_constant_iters
        if anneal_iters is not None:
            self.teacher_kl_anneal_iters = int(anneal_iters)
            self.teacher_kl_cfg["anneal_iters"] = self.teacher_kl_anneal_iters
        if schedule is not None:
            self.teacher_kl_schedule = str(schedule)
            self.teacher_kl_cfg["schedule"] = self.teacher_kl_schedule

    def get_teacher_kl_lambda(self) -> float:
        """Return the current teacher-KL loss weight.

        Warmup takes precedence over all schedules. After warmup, ``constant`` keeps
        ``lambda_start`` fixed, ``linear`` and ``cosine`` anneal toward ``lambda_end``,
        and ``constant_then_linear`` holds ``lambda_start`` before linear annealing.
        """
        if self.teacher_kl_iteration < self.teacher_kl_warmup_iters:
            return 0.0

        schedule = self.teacher_kl_schedule
        if schedule == "constant":
            return self.teacher_kl_lambda_start

        schedule_iteration = self.teacher_kl_iteration - self.teacher_kl_warmup_iters
        if schedule == "constant_then_linear":
            if schedule_iteration < self.teacher_kl_constant_iters:
                return self.teacher_kl_lambda_start
            schedule = "linear"
            schedule_iteration -= self.teacher_kl_constant_iters

        if self.teacher_kl_anneal_iters <= 0:
            return self.teacher_kl_lambda_end

        progress = min(schedule_iteration / self.teacher_kl_anneal_iters, 1.0)
        if schedule == "linear":
            alpha = progress
        elif schedule == "cosine":
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unsupported teacher KL schedule: {self.teacher_kl_schedule}")

        return self.teacher_kl_lambda_start * (1.0 - alpha) + self.teacher_kl_lambda_end * alpha

    def _freeze_teacher(self) -> None:
        """Keep the teacher actor in inference mode and out of gradient updates."""
        if self.teacher is None:
            return

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def load_teacher_checkpoint(self, checkpoint_path: str | None = None, strict: bool = True) -> None:
        """Load the frozen teacher actor from an rsl_rl PPO checkpoint."""
        if self.teacher is None:
            raise RuntimeError("Cannot load a teacher checkpoint before constructing the teacher model.")

        if checkpoint_path is not None:
            self.set_teacher_checkpoint(checkpoint_path)
        if self.teacher_checkpoint_path is None:
            raise ValueError("teacher_kl_cfg.checkpoint_path is required for PPOTeacherKL.")

        loaded_dict = torch.load(self.teacher_checkpoint_path, weights_only=False, map_location=self.device)
        if "actor_state_dict" not in loaded_dict:
            raise KeyError(
                f"Cannot find 'actor_state_dict' in teacher checkpoint: {self.teacher_checkpoint_path}"
            )

        self.teacher.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        self.teacher_loaded = True
        self._teacher_kl_shapes_checked = False
        self._freeze_teacher()
        print(
            "Loaded frozen teacher actor from "
            f"'{self.teacher_checkpoint_path}' with obs groups {self.teacher.obs_groups}."
        )

    def _validate_distribution_params(
        self,
        teacher_params: tuple[torch.Tensor, ...],
        student_params: tuple[torch.Tensor, ...],
    ) -> None:
        """Validate teacher and student distribution parameter compatibility."""
        if len(teacher_params) != len(student_params):
            raise RuntimeError(
                "Teacher/student distribution parameter count mismatch: "
                f"teacher={len(teacher_params)}, student={len(student_params)}"
            )

        for index, (teacher_param, student_param) in enumerate(zip(teacher_params, student_params)):
            if teacher_param.shape != student_param.shape:
                raise RuntimeError(
                    f"Teacher/student distribution parameter {index} shape mismatch: "
                    f"teacher={tuple(teacher_param.shape)}, student={tuple(student_param.shape)}"
                )

        if self.teacher_kl_cfg.get("debug_shapes", False):
            shape_summary = [tuple(param.shape) for param in teacher_params]
            print(f"Teacher/student distribution parameter shapes verified: {shape_summary}")

    def _distributed_mean_scalar(self, value: torch.Tensor) -> torch.Tensor:
        """Average a scalar tensor across distributed workers for logging."""
        value = value.detach()
        if self.is_multi_gpu:
            value = value.clone()
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value /= self.gpu_world_size
        return value

    def _cap_mean_teacher_loss(self, teacher_loss: torch.Tensor) -> torch.Tensor:
        """Apply the optional hard cap for mean-only teacher guidance losses."""
        if self.teacher_guidance_max_loss is None:
            return teacher_loss
        return teacher_loss.clamp(max=self.teacher_guidance_max_loss)

    def _compute_mean_teacher_loss(
        self,
        teacher_params: tuple[torch.Tensor, ...],
        student_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute action-mean teacher guidance losses and diagnostics."""
        teacher_mean = teacher_params[0]
        student_mean = student_params[0]
        if self.teacher_guidance_loss_type == "mean_mse":
            mean_loss = (student_mean - teacher_mean).pow(2).sum(dim=-1).mean()
            loss_key = "teacher_mean_mse"
        elif self.teacher_guidance_loss_type == "mean_huber":
            mean_loss = F.smooth_l1_loss(
                student_mean,
                teacher_mean,
                beta=self.teacher_guidance_huber_delta,
                reduction="none",
            ).sum(dim=-1).mean()
            loss_key = "teacher_mean_huber"
        else:
            raise ValueError(f"Mean teacher loss requested for loss_type={self.teacher_guidance_loss_type}")

        mean_loss_for_update = self._cap_mean_teacher_loss(mean_loss)
        action_mean_l2 = torch.linalg.vector_norm(student_mean.detach() - teacher_mean.detach(), dim=-1).mean()
        teacher_kl = self.actor.get_kl_divergence(
            teacher_params,
            tuple(param.detach() for param in student_params),
        ).mean()

        return mean_loss_for_update, {
            loss_key: mean_loss,
            "teacher_loss_for_update": mean_loss_for_update,
            "teacher_action_mean_l2": action_mean_l2,
            "teacher_kl": teacher_kl,
        }

    def _compute_kl_teacher_loss(
        self,
        teacher_params: tuple[torch.Tensor, ...],
        student_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the legacy full-distribution teacher KL loss."""
        teacher_kl = self.actor.get_kl_divergence(teacher_params, student_params).mean()

        max_kl_loss = self.teacher_kl_cfg.get("max_kl_loss")
        if max_kl_loss is not None:
            max_kl_loss = float(max_kl_loss)
            tail_slope = float(self.teacher_kl_cfg.get("max_kl_loss_tail_slope", 0.0))
            if tail_slope > 0.0:
                teacher_kl_for_loss = torch.where(
                    teacher_kl <= max_kl_loss,
                    teacher_kl,
                    max_kl_loss + tail_slope * (teacher_kl - max_kl_loss),
                )
            else:
                teacher_kl_for_loss = teacher_kl.clamp(max=max_kl_loss)
        else:
            teacher_kl_for_loss = teacher_kl

        return teacher_kl_for_loss, {
            "teacher_kl": teacher_kl,
            "teacher_kl_for_loss": teacher_kl_for_loss,
        }

    def _compute_additional_loss(
        self,
        batch: RolloutStorage.Batch,
        original_batch_size: int,
        distribution_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the frozen-teacher guidance loss."""
        if not self.teacher_guidance_enabled:
            return torch.zeros((), device=self.device), {
                "teacher_loss": 0.0,
                "teacher_loss_for_update": 0.0,
                "teacher_lambda": 0.0,
                "teacher_kl_lambda": 0.0,
                "teacher_guidance_enabled": 0.0,
            }

        if self.teacher is None or not self.teacher_loaded:
            raise RuntimeError("Teacher KL loss requires a loaded teacher model.")
        if batch.observations is None:
            raise RuntimeError("Teacher KL loss requires observations in the rollout batch.")

        teacher_kl_lambda = self.get_teacher_kl_lambda()
        if teacher_kl_lambda == 0.0 and not self.teacher_kl_cfg.get("log_kl_when_lambda_zero", True):
            return torch.zeros((), device=self.device), {
                "teacher_loss": 0.0,
                "teacher_loss_for_update": 0.0,
                "teacher_lambda": 0.0,
                "teacher_kl_lambda": 0.0,
            }

        return self._compute_teacher_guidance_loss(
            batch,
            original_batch_size,
            distribution_params,
            loss_weight=teacher_kl_lambda,
        )

    def _compute_teacher_guidance_loss(
        self,
        batch: RolloutStorage.Batch,
        original_batch_size: int,
        distribution_params: tuple[torch.Tensor, ...],
        loss_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute teacher guidance loss with an explicit scalar weight."""
        if self.teacher is None or not self.teacher_loaded:
            raise RuntimeError("Teacher guidance loss requires a loaded teacher model.")
        if batch.observations is None:
            raise RuntimeError("Teacher guidance loss requires observations in the rollout batch.")

        observations = batch.observations
        teacher_observations = observations[:original_batch_size]
        teacher_masks = batch.masks[:original_batch_size] if batch.masks is not None else None

        with torch.no_grad():
            self.teacher(teacher_observations, masks=teacher_masks, stochastic_output=True)
            teacher_distribution_params = tuple(p.detach() for p in self.teacher.output_distribution_params)

        student_distribution_params = distribution_params
        if loss_weight == 0.0:
            student_distribution_params = tuple(p.detach() for p in student_distribution_params)

        if self.teacher_kl_cfg.get("check_shapes", True) and not self._teacher_kl_shapes_checked:
            self._validate_distribution_params(teacher_distribution_params, student_distribution_params)
            self._teacher_kl_shapes_checked = True

        if self.teacher_guidance_loss_type == "kl":
            teacher_loss_for_update, loss_logs = self._compute_kl_teacher_loss(
                teacher_distribution_params,
                student_distribution_params,
            )
        else:
            teacher_loss_for_update, loss_logs = self._compute_mean_teacher_loss(
                teacher_distribution_params,
                student_distribution_params,
            )

        raw_teacher_kl = loss_logs.get("teacher_kl")
        if (
            raw_teacher_kl is not None
            and self.teacher_kl_cfg.get("fail_on_nonfinite_kl", True)
            and not torch.isfinite(raw_teacher_kl)
        ):
            raise FloatingPointError(
                f"Non-finite teacher KL detected: {raw_teacher_kl.item()}. "
                "Check teacher observation normalization, action distribution parameters, and checkpoint compatibility."
            )
        if not torch.isfinite(teacher_loss_for_update):
            raise FloatingPointError(
                f"Non-finite teacher guidance loss detected: {teacher_loss_for_update.item()}."
            )

        teacher_loss = loss_weight * teacher_loss_for_update
        loss_logs.setdefault("teacher_loss_for_update", teacher_loss_for_update)
        log_dict = {
            name: self._distributed_mean_scalar(value).item()
            for name, value in loss_logs.items()
        }
        teacher_loss_log = self._distributed_mean_scalar(teacher_loss).item()
        log_dict.update(
            {
                "teacher_loss": teacher_loss_log,
                "teacher_lambda": float(loss_weight),
                "teacher_kl_lambda": float(loss_weight),
            }
        )
        if self.teacher_guidance_loss_type == "kl":
            log_dict["teacher_kl_loss"] = teacher_loss_log
        return teacher_loss, log_dict

    def update(self) -> dict[str, float]:
        """Run a PPO update and advance the teacher-KL schedule."""
        if self.teacher_imitation_only:
            loss_dict = self._update_teacher_imitation_only()
        else:
            loss_dict = super().update()
        self.teacher_kl_iteration += 1
        return loss_dict

    def _update_teacher_imitation_only(self) -> dict[str, float]:
        """Run optimization using only frozen-teacher imitation loss."""
        if not self.teacher_guidance_enabled:
            raise RuntimeError("Teacher imitation-only training requires teacher guidance to be enabled.")
        if self.teacher is None or not self.teacher_loaded:
            raise RuntimeError("Teacher imitation-only training requires a loaded teacher model.")
        if self.rnd is not None:
            raise NotImplementedError("Teacher imitation-only training does not support RND.")
        if self.symmetry is not None:
            raise NotImplementedError("Teacher imitation-only training does not support symmetry augmentation.")

        mean_entropy = 0.0
        mean_teacher_losses: dict[str, float] = {}

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if batch.observations is None:
                raise RuntimeError("Teacher imitation-only training requires observations in the rollout batch.")
            original_batch_size = batch.observations.batch_size[0]

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            loss, loss_logs = self._compute_teacher_guidance_loss(
                batch,
                original_batch_size,
                distribution_params,
                loss_weight=self.teacher_imitation_loss_coef,
            )

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_entropy += entropy.mean().item()
            for name, value in loss_logs.items():
                mean_teacher_losses[name] = mean_teacher_losses.get(name, 0.0) + value

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_entropy /= num_updates
        for name in mean_teacher_losses:
            mean_teacher_losses[name] /= num_updates

        loss_dict = {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": mean_entropy,
            "teacher_imitation_only": 1.0,
        }
        loss_dict.update(mean_teacher_losses)
        self.storage.clear()
        return loss_dict

    def save(self) -> dict:
        """Return a dict of all learnable models and teacher-KL training state."""
        saved_dict = super().save()
        saved_dict["teacher_kl_iteration"] = self.teacher_kl_iteration
        saved_dict["teacher_kl_cfg"] = dict(self.teacher_kl_cfg)
        saved_dict["teacher_checkpoint_path"] = self.teacher_checkpoint_path
        saved_dict["teacher_kl_lambda_current"] = self.get_teacher_kl_lambda()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load models and restore the teacher-KL schedule state."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration and "teacher_kl_iteration" in loaded_dict:
            self.teacher_kl_iteration = int(loaded_dict["teacher_kl_iteration"])
        return load_iteration

    def train_mode(self) -> None:
        """Set train mode for learnable models while keeping the teacher frozen."""
        super().train_mode()
        self._freeze_teacher()

    def eval_mode(self) -> None:
        """Set evaluation mode for all policy models."""
        super().eval_mode()
        self._freeze_teacher()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOTeacherKL:
        """Construct the PPO + teacher-KL algorithm."""
        algorithm_cfg = copy.deepcopy(cfg["algorithm"])
        actor_cfg = copy.deepcopy(cfg["actor"])
        critic_cfg = copy.deepcopy(cfg["critic"])
        teacher_cfg = copy.deepcopy(cfg.get("teacher", cfg["actor"]))
        obs_groups_cfg = copy.deepcopy(cfg["obs_groups"])
        teacher_guidance_cfg = algorithm_cfg.get("teacher_kl_cfg") or {}
        teacher_guidance_enabled = bool(teacher_guidance_cfg.get("enabled", True))

        # Resolve class callables
        alg_class: type[PPOTeacherKL] = resolve_callable(algorithm_cfg.pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore
        if teacher_guidance_enabled:
            teacher_class_name = teacher_cfg.pop("class_name", None)
            teacher_class: type[MLPModel] | None = (
                resolve_callable(teacher_class_name) if teacher_class_name else actor_class
            )
        else:
            teacher_class = None

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if teacher_guidance_enabled:
            default_sets.append("teacher")
        if "rnd_cfg" in algorithm_cfg and algorithm_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        obs_groups = resolve_obs_groups(obs, obs_groups_cfg, default_sets)

        # Resolve RND config if used
        algorithm_cfg = resolve_rnd_config(algorithm_cfg, obs, obs_groups, env)

        # Resolve symmetry config if used
        algorithm_cfg = resolve_symmetry_config(algorithm_cfg, env)

        # Initialize the student actor, critic, and frozen teacher actor
        actor: MLPModel = actor_class(obs, obs_groups, "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        if algorithm_cfg.pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            critic_cfg["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")
        teacher: MLPModel | None = None
        if teacher_guidance_enabled:
            if teacher_class is None:
                raise RuntimeError("teacher_class should be resolved when teacher guidance is enabled.")
            teacher = teacher_class(obs, obs_groups, "teacher", env.num_actions, **teacher_cfg).to(device)
            if teacher.is_recurrent:
                raise ValueError("PPOTeacherKL currently supports feedforward teacher actors only.")
            print(f"Teacher Model: {teacher}")
        else:
            print("Teacher guidance disabled: training with PPO loss only.")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm. The teacher is intentionally assigned after construction so it is not part of the
        # PPO optimizer created by the parent class.
        alg: PPOTeacherKL = alg_class(
            actor, critic, storage, device=device, **algorithm_cfg, multi_gpu_cfg=cfg["multi_gpu"]
        )
        if teacher is not None:
            alg.teacher = teacher
            alg.load_teacher_checkpoint()

        # Compile the algorithm's learnable models if requested. The teacher remains an uncompiled frozen reference.
        alg.compile(cfg.get("torch_compile_mode"))

        return alg
