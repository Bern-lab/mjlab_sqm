"""Small dependency-light PCA helper for collected latent datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass(frozen=True)
class AnalyzeLatentsConfig:
  input_file: str
  output_dir: str | None = None
  output_file: str | None = None
  plot_file: str | None = None
  phase_bins: int | None = None
  phase_bin: int | None = None
  phase_plot_dir: str | None = None
  write_plot: bool = True
  max_samples: int | None = 50000
  seed: int = 34567


@dataclass(frozen=True)
class AnalyzeOutputPaths:
  output_file: Path
  plot_file: Path | None
  phase_plot_dir: Path | None


def _resolve_output_paths(cfg: AnalyzeLatentsConfig) -> AnalyzeOutputPaths:
  input_path = Path(cfg.input_file)
  output_dir = (
    Path(cfg.output_dir) if cfg.output_dir is not None else input_path.parent
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  output_file = (
    Path(cfg.output_file)
    if cfg.output_file is not None
    else output_dir / "latent_pca.csv"
  )
  if cfg.plot_file is not None:
    plot_file = Path(cfg.plot_file)
  elif cfg.write_plot:
    plot_file = output_dir / "latent_pca.png"
  else:
    plot_file = None

  if cfg.phase_plot_dir is not None:
    phase_plot_dir = Path(cfg.phase_plot_dir)
  elif cfg.phase_bins is not None:
    phase_plot_dir = output_dir / "phase_bins"
  else:
    phase_plot_dir = None

  output_file.parent.mkdir(parents=True, exist_ok=True)
  if plot_file is not None:
    plot_file.parent.mkdir(parents=True, exist_ok=True)
  if phase_plot_dir is not None:
    phase_plot_dir.mkdir(parents=True, exist_ok=True)
  return AnalyzeOutputPaths(
    output_file=output_file,
    plot_file=plot_file,
    phase_plot_dir=phase_plot_dir,
  )


def _compute_phase_bins(phase: np.ndarray, num_bins: int) -> np.ndarray:
  return np.floor((phase % 1.0) * num_bins).astype(np.int64).clip(0, num_bins - 1)


def _plot_pca(
  *,
  pcs: np.ndarray,
  labels: np.ndarray,
  explained: np.ndarray,
  output_path: Path,
  title: str,
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=160)
  unique_labels = np.unique(labels)
  cmap = plt.get_cmap("tab10")
  for label_idx, label in enumerate(unique_labels):
    mask = labels == label
    ax.scatter(
      pcs[mask, 0],
      pcs[mask, 1],
      s=6,
      alpha=0.45,
      color=cmap(label_idx % cmap.N),
      linewidths=0,
      label=str(label),
    )
  ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
  ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
  ax.set_title(title)
  ax.legend(markerscale=3.0, frameon=False, fontsize=8)
  ax.grid(True, alpha=0.2)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def run_analyze_latents(cfg: AnalyzeLatentsConfig) -> None:
  output_paths = _resolve_output_paths(cfg)
  print(f"[INFO] Output directory: {output_paths.output_file.parent}")
  data = np.load(cfg.input_file, allow_pickle=False)
  latent = np.asarray(data["latent"], dtype=np.float32)
  phase = (
    np.asarray(data["gait_phase"], dtype=np.float32)
    if "gait_phase" in data
    else None
  )
  n = latent.shape[0]
  if cfg.max_samples is not None and n > cfg.max_samples:
    rng = np.random.default_rng(cfg.seed)
    idx = np.sort(rng.choice(n, size=cfg.max_samples, replace=False))
    latent = latent[idx]
    labels = data["terrain_label"][idx]
    names = data["terrain_name"][idx]
    heights = data["terrain_height_m"][idx]
    time_steps = data["time_step"][idx]
    if phase is not None:
      phase = phase[idx]
  else:
    labels = data["terrain_label"]
    names = data["terrain_name"]
    heights = data["terrain_height_m"]
    time_steps = data["time_step"]

  phase_bin_ids = None
  if cfg.phase_bins is not None:
    if phase is None:
      raise ValueError(
        "This latent file has no gait_phase array. Re-run collect_policy_latents.py."
      )
    if cfg.phase_bins <= 0:
      raise ValueError("--phase-bins must be positive.")
    phase_bin_ids = _compute_phase_bins(phase, cfg.phase_bins)
    if cfg.phase_bin is not None:
      if cfg.phase_bin < 0 or cfg.phase_bin >= cfg.phase_bins:
        raise ValueError("--phase-bin must be in [0, phase_bins).")
      mask = phase_bin_ids == cfg.phase_bin
      latent = latent[mask]
      labels = labels[mask]
      names = names[mask]
      heights = heights[mask]
      time_steps = time_steps[mask]
      phase = phase[mask]
      phase_bin_ids = phase_bin_ids[mask]

  centered = latent - latent.mean(axis=0, keepdims=True)
  _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
  pcs = centered @ vt[:2].T
  variance = singular_values**2
  explained = variance[:2] / max(float(variance.sum()), 1e-12)

  output_path = output_paths.output_file
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["pc1", "pc2", "terrain_label", "terrain_name", "height_m", "time_step"]
    if phase is not None:
      header.append("gait_phase")
    if phase_bin_ids is not None:
      header.append("phase_bin")
    writer.writerow(header)
    for row_idx in range(pcs.shape[0]):
      row = [
        float(pcs[row_idx, 0]),
        float(pcs[row_idx, 1]),
        str(labels[row_idx]),
        str(names[row_idx]),
        float(heights[row_idx]),
        int(time_steps[row_idx]),
      ]
      if phase is not None:
        row.append(float(phase[row_idx]))
      if phase_bin_ids is not None:
        row.append(int(phase_bin_ids[row_idx]))
      writer.writerow(row)
  print(f"[INFO] Wrote PCA CSV to {output_path}")

  if output_paths.plot_file is not None:
    plot_path = output_paths.plot_file
    title = "Actor latent PCA by eval terrain"
    if cfg.phase_bin is not None and cfg.phase_bins is not None:
      title += f" | phase bin {cfg.phase_bin}/{cfg.phase_bins}"
    _plot_pca(
      pcs=pcs,
      labels=labels,
      explained=explained,
      output_path=plot_path,
      title=title,
    )
    print(f"[INFO] Wrote PCA plot to {plot_path}")

  if output_paths.phase_plot_dir is not None:
    if phase_bin_ids is None or cfg.phase_bins is None:
      raise ValueError("Use --phase-bins when providing --phase-plot-dir.")
    phase_dir = output_paths.phase_plot_dir
    for bin_idx in range(cfg.phase_bins):
      mask = phase_bin_ids == bin_idx
      if not np.any(mask):
        continue
      start = bin_idx / cfg.phase_bins
      end = (bin_idx + 1) / cfg.phase_bins
      phase_path = phase_dir / f"phase_{bin_idx:02d}_{start:.2f}_{end:.2f}.png"
      _plot_pca(
        pcs=pcs[mask],
        labels=labels[mask],
        explained=explained,
        output_path=phase_path,
        title=f"Actor latent PCA | gait phase [{start:.2f}, {end:.2f})",
      )
    print(f"[INFO] Wrote phase-bin PCA plots to {phase_dir}")


def main() -> None:
  cfg = tyro.cli(AnalyzeLatentsConfig)
  run_analyze_latents(cfg)


if __name__ == "__main__":
  main()
