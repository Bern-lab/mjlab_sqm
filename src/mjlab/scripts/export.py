"""Script to export a trained RL policy to TorchScript (.pt) format."""

import argparse
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import (
    get_checkpoint_path_with_fallback,
    get_task_log_root,
)


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_file: str | None = None
    """Path to the checkpoint file to export."""
    load_run: str = ".*"
    """Run directory pattern used when checkpoint_file is not provided."""
    load_checkpoint: str = "model_.*.pt"
    """Checkpoint filename pattern used when checkpoint_file is not provided."""
    motion_file: str | None = None
    """Path to motion file for tracking tasks."""
    output_dir: str | None = None
    """Output directory for exported model. Defaults to <checkpoint_dir>/exported."""
    output_name: str | None = None
    """Exported TorchScript file name. Defaults to <run_dir>.pt."""
    device: str = "cuda:0"
    """Device for loading checkpoint and building environment."""


def load_motion_file_from_log(checkpoint_dir: Path) -> str | None:
    """Load motion_file from params/env.yaml in a checkpoint directory."""
    env_yaml_path = checkpoint_dir / "params" / "env.yaml"
    if not env_yaml_path.exists():
        return None

    try:
        content = env_yaml_path.read_text(encoding="utf-8")
        match = re.search(r"motion_file:\s*(.+\.npz)", content)
        if match:
            motion_file = match.group(1).strip()
            if Path(motion_file).exists():
                return motion_file
    except Exception as exc:
        print(f"[WARN]: Failed to parse motion file from env.yaml: {exc}")

    return None


def _checkpoint_sort_key(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


def resolve_checkpoint_path(checkpoint_arg: str, log_root: Path) -> tuple[Path, Path]:
    """Resolve user checkpoint argument to (log_dir, checkpoint_file)."""
    checkpoint_path = Path(checkpoint_arg)

    if checkpoint_path.is_absolute() and checkpoint_path.exists():
        if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
            return checkpoint_path.parent, checkpoint_path
        if checkpoint_path.is_dir():
            model_files = sorted(checkpoint_path.glob("model_*.pt"), key=_checkpoint_sort_key)
            if not model_files:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
            return checkpoint_path, model_files[-1]

    if checkpoint_path.exists():
        if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
            return checkpoint_path.parent, checkpoint_path
        if checkpoint_path.is_dir():
            model_files = sorted(checkpoint_path.glob("model_*.pt"), key=_checkpoint_sort_key)
            if not model_files:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
            return checkpoint_path, model_files[-1]

    full_path = log_root / checkpoint_arg
    if full_path.exists():
        if full_path.is_file() and full_path.suffix == ".pt":
            return full_path.parent, full_path
        if full_path.is_dir():
            model_files = sorted(full_path.glob("model_*.pt"), key=_checkpoint_sort_key)
            if not model_files:
                raise FileNotFoundError(f"No checkpoint files found in {full_path}")
            return full_path, model_files[-1]

    matching_dirs = [d for d in log_root.glob(f"**/{checkpoint_arg}") if d.is_dir()]
    if matching_dirs:
        log_dir = matching_dirs[0]
        model_files = sorted(log_dir.glob("model_*.pt"), key=_checkpoint_sort_key)
        if not model_files:
            raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
        return log_dir, model_files[-1]

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")


def resolve_checkpoint_path_with_fallback(
    checkpoint_arg: str, log_roots: list[Path]
) -> tuple[Path, Path]:
    """Resolve a checkpoint by searching multiple log roots in order."""
    errors: list[str] = []
    for log_root in log_roots:
        try:
            return resolve_checkpoint_path(checkpoint_arg, log_root)
        except FileNotFoundError as exc:
            errors.append(f"{log_root}: {exc}")
    joined = "\n".join(errors)
    raise FileNotFoundError(
        f"Checkpoint not found under any candidate log root:\n{joined}"
    )


def export_policy(task: str, cfg: ExportConfig) -> None:
    """Export trained policy to deployable TorchScript format."""
    env_cfg = load_env_cfg(task)
    agent_cfg = load_rl_cfg(task)
    assert isinstance(agent_cfg, RslRlBaseRunnerCfg)

    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    else:
        log_root_path = get_task_log_root(agent_cfg.experiment_name, task)
        legacy_log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
        resume_path = get_checkpoint_path_with_fallback(
            [log_root_path, legacy_log_root_path],
            cfg.load_run,
            cfg.load_checkpoint,
        )

    log_dir = resume_path.parent

    if "motion" in env_cfg.commands and hasattr(env_cfg.commands["motion"], "motion_file"):
        motion_file = cfg.motion_file
        if motion_file is None:
            motion_file = load_motion_file_from_log(log_dir)
            if motion_file is not None:
                print(f"[INFO]: Loaded motion file from log directory: {motion_file}")
        if motion_file is None:
            raise ValueError(
                "Motion file required for tracking tasks. Provide --motion-file or ensure params/env.yaml contains motion_file."
            )
        if not Path(motion_file).exists():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        env_cfg.commands["motion"].motion_file = motion_file
        print(f"[INFO]: Using motion file: {motion_file}")

    env_cfg.sim.nconmax = max(env_cfg.sim.nconmax, 512)
    env_cfg.sim.njmax = max(env_cfg.sim.njmax, 4096)

    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = log_dir / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output_name is not None:
        output_name = cfg.output_name
    else:
        output_name = f"{log_dir.name}.pt"

    print(f"[INFO]: Loading checkpoint: {resume_path}")
    print(f"[INFO]: Export directory: {output_dir}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=cfg.device)
        runner.load(str(resume_path), map_location=cfg.device)
        runner.export_policy_to_jit(str(output_dir), output_name)
        print(f"[INFO]: Exported TorchScript model to: {output_dir / output_name}")
    finally:
        env.close()


def _parse_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    try:
        return f"cuda:{int(device_arg)}"
    except ValueError:
        return device_arg


def _run_simplified_interface() -> None:
    parser = argparse.ArgumentParser(
        description="Simplified export interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python src/mjlab/scripts/export.py -c k1_run -t Mjlab-Tracking-Flat-Booster_K1-No-State-Estimation
    uv run python src/mjlab/scripts/export.py -c k1_run/model_5000.pt -t Mjlab-Tracking-Flat-Booster_K1-No-State-Estimation
    uv run python src/mjlab/scripts/export.py -c k1_run -t Mjlab-Tracking-Flat-Booster_K1-No-State-Estimation --motion data/k1/motion.npz
        """,
    )
    parser.add_argument("-c", "--checkpoint", required=True, help="Run name, run dir, or checkpoint .pt path")
    parser.add_argument("-t", "--task", required=True, help="Task name")
    parser.add_argument("-d", "--device", default="0", help="CUDA id (e.g. 0) or cpu")
    parser.add_argument("--motion", default=None, help="Motion file override path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--output-name", default=None, help="Output .pt filename")
    args = parser.parse_args()

    all_tasks = set(list_tasks())
    if args.task not in all_tasks:
        raise ValueError(f"Unknown task: {args.task}")

    agent_cfg = load_rl_cfg(args.task)
    assert isinstance(agent_cfg, RslRlBaseRunnerCfg)
    log_root_path = get_task_log_root(agent_cfg.experiment_name, args.task)
    legacy_log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name

    log_dir, checkpoint_file = resolve_checkpoint_path_with_fallback(
        args.checkpoint, [log_root_path, legacy_log_root_path]
    )
    print(f"[INFO]: Checkpoint: {checkpoint_file}")
    print(f"[INFO]: Log directory: {log_dir}")

    motion_file: str | None = args.motion
    if motion_file is None:
        motion_file = load_motion_file_from_log(log_dir)
        if motion_file is not None:
            print(f"[INFO]: Loaded motion file from log directory: {motion_file}")

    export_cfg = ExportConfig(
        checkpoint_file=str(checkpoint_file),
        motion_file=motion_file,
        output_dir=args.output_dir,
        output_name=args.output_name,
        device=_parse_device(args.device),
    )
    export_policy(args.task, export_cfg)


def _run_full_interface() -> None:
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        ExportConfig,
        args=remaining_args,
        default=ExportConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    export_policy(chosen_task, args)


def main() -> None:
    import mjlab.tasks  # noqa: F401

    use_simplified = False
    if len(sys.argv) > 1:
        if "-c" in sys.argv or "--checkpoint" in sys.argv:
            use_simplified = True
        elif "--checkpoint-file" in sys.argv:
            use_simplified = False
        elif sys.argv[1].startswith("-"):
            use_simplified = True

    try:
        if use_simplified:
            _run_simplified_interface()
        else:
            _run_full_interface()
    except KeyboardInterrupt:
        print("\n[INFO]: Export interrupted by user (Ctrl+C). Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
