"""Clean training log artifacts produced by RSL-RL runs."""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

SECONDS_PER_DAY = 86400
MB = 1024 * 1024


@dataclass(frozen=True)
class FileKind:
  key: str
  label: str
  pattern: str


@dataclass
class FileStats:
  deleted: int = 0
  to_delete: int = 0
  size_mb: float = 0.0


FILE_KINDS = {
  "checkpoints": FileKind("checkpoints", "Checkpoints", "model_*.pt"),
  "models": FileKind("models", "Models", "exported/*.pt"),
  "videos": FileKind("videos", "Videos", "videos/play/*.mp4"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="清理训练日志文件",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
使用示例:
  # 删除3天前的checkpoints、视频和导出模型
  uv run clean -c -v -m -d 3

  # 删除3天前的checkpoints和视频，但保留所有导出模型
  uv run clean -c -v -d 3

  # 只删除7天前的视频
  uv run clean -v -d 7

  # 每个实验目录只保留最新的一个checkpoint、视频和模型
  uv run clean --all

  # 先预览，不实际删除
  uv run clean --all --dry-run
    """,
  )

  parser.add_argument(
    "-c",
    "--checkpoints",
    action="store_true",
    help="清理checkpoint文件 (model_*.pt)",
  )
  parser.add_argument(
    "-v",
    "--videos",
    action="store_true",
    help="清理视频文件 (videos/play/*.mp4)",
  )
  parser.add_argument(
    "-m",
    "--models",
    action="store_true",
    help="清理导出的模型文件 (exported/*.pt)",
  )
  parser.add_argument(
    "-d",
    "--days",
    type=int,
    default=None,
    help="保留最近N天的文件，删除N天前的文件",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="每个实验目录只保留最新的一个文件；未指定-c/-v/-m时默认处理所有类型",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="预览模式：只显示将要删除的文件，不实际删除",
  )
  parser.add_argument(
    "--log-dir",
    type=Path,
    default=Path("logs/rsl_rl"),
    help="日志根目录 (默认: logs/rsl_rl)",
  )

  return parser.parse_args(argv)


def get_file_age_days(file_path: Path, now: float | None = None) -> float:
  """Return the file age in days."""
  now = time.time() if now is None else now
  return (now - file_path.stat().st_mtime) / SECONDS_PER_DAY


def _format_size(path: Path) -> float:
  return path.stat().st_size / MB


def _has_readable_zip_directory(path: Path) -> bool:
  try:
    with zipfile.ZipFile(path) as archive:
      archive.namelist()
  except (OSError, zipfile.BadZipFile):
    return False
  return True


def _iter_run_dir_candidates(log_root: Path) -> Iterable[Path]:
  yield log_root

  for exp_name_dir in sorted(log_root.iterdir()):
    if not exp_name_dir.is_dir():
      continue

    yield exp_name_dir
    for run_dir in sorted(exp_name_dir.iterdir()):
      if run_dir.is_dir():
        yield run_dir


def _looks_like_run_dir(path: Path) -> bool:
  return (
    "-Exp-" in path.name
    or any(path.glob("model_*.pt"))
    or (path / "exported").is_dir()
    or (path / "videos" / "play").is_dir()
  )


def find_experiment_dirs(log_root: Path) -> list[Path]:
  """Find run directories under logs/rsl_rl.

  Current mjlab training writes logs to:
    logs/rsl_rl/{experiment_name}/{timestamp_or_run_name}

  Older/local workflows may use directories containing "-Exp-", so those are
  still accepted.
  """
  if not log_root.exists():
    print(f"[WARN] 日志目录不存在: {log_root}")
    return []
  if not log_root.is_dir():
    print(f"[WARN] 日志路径不是目录: {log_root}")
    return []

  exp_dirs = {
    candidate
    for candidate in _iter_run_dir_candidates(log_root)
    if candidate.is_dir() and _looks_like_run_dir(candidate)
  }
  return sorted(exp_dirs)


def _selected_file_kinds(args: argparse.Namespace, default_all: bool) -> list[FileKind]:
  if default_all and not any([args.checkpoints, args.videos, args.models]):
    return list(FILE_KINDS.values())

  selected = []
  if args.checkpoints:
    selected.append(FILE_KINDS["checkpoints"])
  if args.models:
    selected.append(FILE_KINDS["models"])
  if args.videos:
    selected.append(FILE_KINDS["videos"])
  return selected


def _record_file(
  file_path: Path,
  stats: FileStats,
  dry_run: bool,
  now: float,
) -> None:
  age = get_file_age_days(file_path, now)
  size_mb = _format_size(file_path)
  print(f"    - {file_path.name} ({age:.1f}天, {size_mb:.1f}MB)")

  if dry_run:
    stats.to_delete += 1
  else:
    file_path.unlink()
    stats.deleted += 1
  stats.size_mb += size_mb


def _select_latest_file(files: list[Path], kind: FileKind) -> Path:
  if kind.key != "checkpoints":
    return files[-1]

  readable_files = [f for f in files if _has_readable_zip_directory(f)]
  if readable_files:
    return readable_files[-1]

  print("    [WARN] 未找到可读 checkpoint，按修改时间保留最新文件")
  return files[-1]


def clean_by_days(
  exp_dirs: list[Path],
  args: argparse.Namespace,
  file_stats: dict[str, FileStats],
) -> None:
  """Clean files older than args.days."""
  assert args.days is not None
  cutoff_days = args.days
  kinds = _selected_file_kinds(args, default_all=False)
  now = time.time()

  print(f"\n{'=' * 80}")
  print(f"清理模式: 删除 {cutoff_days} 天前的文件")
  print(f"{'=' * 80}\n")

  for exp_dir in exp_dirs:
    print(f"检查实验目录: {exp_dir}")

    for kind in kinds:
      files = [f for f in exp_dir.glob(kind.pattern) if f.is_file()]
      old_files = [f for f in files if get_file_age_days(f, now) > cutoff_days]
      if not old_files:
        continue

      print(f"  [{kind.label}] 找到 {len(old_files)} 个文件 (>{cutoff_days}天)")
      for file_path in sorted(old_files, key=lambda f: f.stat().st_mtime):
        _record_file(file_path, file_stats[kind.key], args.dry_run, now)


def clean_keep_latest(
  exp_dirs: list[Path],
  args: argparse.Namespace,
  file_stats: dict[str, FileStats],
) -> None:
  """Keep only the newest file of each selected kind in every run directory."""
  kinds = _selected_file_kinds(args, default_all=True)
  now = time.time()

  print(f"\n{'=' * 80}")
  print("清理模式: 每个实验目录每类文件只保留最新的一个")
  print(f"{'=' * 80}\n")

  for exp_dir in exp_dirs:
    print(f"检查实验目录: {exp_dir}")

    for kind in kinds:
      files = sorted(
        (f for f in exp_dir.glob(kind.pattern) if f.is_file()),
        key=lambda f: f.stat().st_mtime,
      )
      if len(files) <= 1:
        continue

      latest = _select_latest_file(files, kind)
      to_delete = [f for f in files if f != latest]
      print(f"  [{kind.label}] 共 {len(files)} 个，保留最新，删除 {len(to_delete)} 个")
      for file_path in to_delete:
        _record_file(file_path, file_stats[kind.key], args.dry_run, now)
      print(f"    [KEEP] {latest.name}")


def _validate_args(args: argparse.Namespace) -> bool:
  if not args.all and not any([args.checkpoints, args.videos, args.models]):
    print("[ERROR] 请至少指定一个清理类型: -c, -v, -m, 或 --all")
    return False

  if args.all and args.days is not None:
    print("[ERROR] --all 和 -d/--days 不能同时使用")
    return False

  if not args.all and args.days is None:
    print("[ERROR] 使用 -c/-v/-m 时必须指定 -d/--days 参数")
    return False

  if args.days is not None and args.days < 0:
    print("[ERROR] -d/--days 必须大于或等于 0")
    return False

  return True


def _print_summary(
  file_stats: dict[str, FileStats],
  dry_run: bool,
) -> None:
  print(f"\n{'=' * 80}")
  print("清理总结")
  print(f"{'=' * 80}")

  if dry_run:
    total = sum(stat.to_delete for stat in file_stats.values())
    total_size = sum(stat.size_mb for stat in file_stats.values())
    print(f"预计删除文件总数: {total}")
    print(f"预计释放磁盘空间: {total_size:.1f} MB ({total_size / 1024:.2f} GB)")
    for kind in FILE_KINDS.values():
      stat = file_stats[kind.key]
      if stat.to_delete > 0:
        print(f"  - {kind.label}: {stat.to_delete} 个 ({stat.size_mb:.1f} MB)")
    print("\n使用相同参数但去掉 --dry-run 来实际执行删除")
  else:
    total = sum(stat.deleted for stat in file_stats.values())
    total_size = sum(stat.size_mb for stat in file_stats.values())
    print(f"已删除文件总数: {total}")
    print(f"释放磁盘空间: {total_size:.1f} MB ({total_size / 1024:.2f} GB)")
    for kind in FILE_KINDS.values():
      stat = file_stats[kind.key]
      if stat.deleted > 0:
        print(f"  - {kind.label}: {stat.deleted} 个 ({stat.size_mb:.1f} MB)")

  print(f"{'=' * 80}\n")


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  if not _validate_args(args):
    return 1

  file_stats = {
    "checkpoints": FileStats(),
    "models": FileStats(),
    "videos": FileStats(),
  }

  if args.dry_run:
    print(f"\n{'=' * 80}")
    print("预览模式: 不会实际删除文件")
    print(f"{'=' * 80}\n")

  exp_dirs = find_experiment_dirs(args.log_dir)
  if not exp_dirs:
    print(f"[INFO] 在 {args.log_dir} 下未找到任何实验目录")
    return 0

  print(f"[INFO] 找到 {len(exp_dirs)} 个实验目录\n")

  if args.all:
    clean_keep_latest(exp_dirs, args, file_stats)
  else:
    clean_by_days(exp_dirs, args, file_stats)

  _print_summary(file_stats, args.dry_run)
  return 0


if __name__ == "__main__":
  sys.exit(main())
