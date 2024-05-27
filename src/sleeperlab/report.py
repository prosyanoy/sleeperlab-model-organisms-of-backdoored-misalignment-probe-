"""Aggregate per-seed records into a results table (markdown + CSV)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

METRICS = ["asr_trigger", "asr_no_trigger", "clean_acc"]


def aggregate(records: list[dict]) -> list[dict]:
    """Mean/std over seeds for each (safety_lr, safety_steps)."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["safety_lr"], r["safety_steps"]), []).append(r)
    rows = []
    for (lr, steps), rs in sorted(groups.items()):
        row = {"safety_lr": lr, "safety_steps": steps, "n_seeds": len(rs)}
        for m in METRICS:
            v = np.array([r[m] for r in rs], dtype=float)
            row[m + "_mean"], row[m + "_std"] = float(np.mean(v)), float(np.std(v))
        for key in ("auroc_trigger", "auroc_defect", "auroc_trigger_transfer"):
            arr = np.array([r[key] for r in rs], dtype=float)  # [seeds, layers]
            if np.isnan(arr).all():
                row[key + "_best"], row[key + "_best_layer"] = float("nan"), -1
                continue
            mean = np.nanmean(arr, axis=0)
            row[key + "_best_layer"] = int(np.nanargmax(mean))
            row[key + "_best"] = float(mean[row[key + "_best_layer"]])
        rows.append(row)
    return rows


def write_table(records: list[dict], out: Path) -> None:
    rows = aggregate(records)
    with open(out / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pm = lambda r, m: f"{r[m + '_mean']:.2f} ± {r[m + '_std']:.2f}"
    fmt = lambda v: "n/a" if np.isnan(v) else f"{v:.3f}"
    lines = ["| safety lr | safety steps | seeds | ASR (trigger) | ASR (no trigger) | clean acc "
             "| AUROC trigger (best layer) | AUROC defect (best layer) | AUROC transfer (best layer) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['safety_lr']:g} | {r['safety_steps']} | {r['n_seeds']} | "
                     f"{pm(r, 'asr_trigger')} | {pm(r, 'asr_no_trigger')} | {pm(r, 'clean_acc')} | "
                     + " | ".join(f"{fmt(r[k + '_best'])} (L{r[k + '_best_layer']})"
                                  for k in ("auroc_trigger", "auroc_defect", "auroc_trigger_transfer"))
                     + " |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")
