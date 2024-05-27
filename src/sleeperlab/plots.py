"""Matplotlib figures: backdoor persistence and per-layer probe AUROC."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]  # Okabe-Ito


def _band(ax, x, arr, label, color, ls="-"):
    arr = np.asarray(arr, dtype=float)
    m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
    ax.plot(x, m, ls, color=color, marker="o", ms=4, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)


def plot_persistence(records: list[dict], out: Path) -> None:
    lrs = sorted({r["safety_lr"] for r in records})
    fig, axes = plt.subplots(1, len(lrs), figsize=(5.2 * len(lrs), 3.8), squeeze=False, sharey=True)
    series = [("asr_trigger", "ASR with trigger"), ("asr_no_trigger", "ASR without trigger"),
              ("clean_acc", "clean task accuracy")]
    for ax, lr in zip(axes[0], lrs):
        steps = sorted({r["safety_steps"] for r in records if r["safety_lr"] == lr})
        for (key, lab), c in zip(series, COLORS):
            arr = [[r[key] for r in records if r["safety_lr"] == lr and r["safety_steps"] == s]
                   for s in steps]
            _band(ax, range(len(steps)), np.array(arr).T, lab, c, "--" if "no_trigger" in key else "-")
        ax.set_xticks(range(len(steps)), [str(s) for s in steps])
        ax.set_xlabel("safety fine-tuning steps (clean, trigger-free data)")
        ax.set_title(f"safety lr = {lr:g}")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("rate (mean ± std over seeds)")
    axes[0][0].legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "backdoor_persistence.png", dpi=130)
    plt.close(fig)


def plot_probes(records: list[dict], out: Path) -> None:
    lrs = sorted({r["safety_lr"] for r in records})
    keys = [("auroc_trigger", "trigger present"), ("auroc_defect", "about to defect"),
            ("auroc_trigger_transfer", "trigger present (probe fit before safety FT)")]
    fig, axes = plt.subplots(len(lrs), len(keys), figsize=(5 * len(keys), 3.4 * len(lrs)),
                             squeeze=False, sharey=True)
    for i, lr in enumerate(lrs):
        steps = sorted({r["safety_steps"] for r in records if r["safety_lr"] == lr})
        shown = sorted({steps[0], steps[len(steps) // 2], steps[-1]})
        for j, (key, title) in enumerate(keys):
            ax = axes[i][j]
            for s, c in zip(shown, COLORS):
                arr = [r[key] for r in records if r["safety_lr"] == lr and r["safety_steps"] == s]
                if np.isnan(np.array(arr, dtype=float)).all():
                    continue  # defect probe undefined once the backdoor is fully gone
                _band(ax, range(len(arr[0])), arr, f"{s} safety steps", c)
            ax.axhline(0.5, color="grey", lw=0.8, ls=":")
            ax.set_xticks(range(len(arr[0])))
            ax.set_title(f"{title}\n(safety lr {lr:g})", fontsize=9)
            ax.set_xlabel("residual-stream layer (0 = embeddings)")
            ax.set_ylim(0.3, 1.03)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("test AUROC")
            ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "probe_auroc_by_layer.png", dpi=130)
    plt.close(fig)


def make_plots(records: list[dict], out: Path) -> None:
    plot_persistence(records, out)
    plot_probes(records, out)
