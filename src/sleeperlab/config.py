"""Experiment profiles; env vars supply local defaults (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    seeds: list[int]
    safety_steps: list[int]
    safety_lrs: list[float]
    backdoor_steps: int
    backdoor_lr: float = 2e-3
    trigger_frac: float = 0.5
    batch_size: int = 64
    safety_batch_size: int = 64
    d_model: int = 64
    n_layer: int = 4
    n_eval: int = 600
    n_probe_train: int = 1000
    n_probe_test: int = 600
    out_dir: str = "results"
    device: str = "cpu"
    name: str = "full"


PROFILES = {
    "smoke": dict(seeds=[0], safety_steps=[0, 20], safety_lrs=[3e-4], backdoor_steps=300,
                  n_eval=100, n_probe_train=200, n_probe_test=100),
    "demo": dict(seeds=[0, 1], safety_steps=[0, 50, 200], safety_lrs=[3e-4, 3e-3],
                 backdoor_steps=600, n_eval=200, n_probe_train=400, n_probe_test=200),
    "quick": dict(seeds=[0, 1], safety_steps=[0, 50, 200], safety_lrs=[3e-4, 3e-3],
                  backdoor_steps=1500, n_eval=300, n_probe_train=500, n_probe_test=300),
    "full": dict(seeds=[0, 1, 2, 3, 4], safety_steps=[0, 50, 100, 200, 400, 800, 1600],
                 safety_lrs=[3e-4, 3e-3], backdoor_steps=3000),
}


def get_config(profile: str | None = None, out_dir: str | None = None) -> Config:
    profile = profile or os.environ.get("SLEEPERLAB_PROFILE", "full")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    return Config(**PROFILES[profile], name=profile,
                  out_dir=out_dir or os.environ.get("SLEEPERLAB_OUT", "results"),
                  device=os.environ.get("SLEEPERLAB_DEVICE", "cpu"))
