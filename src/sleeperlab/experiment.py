"""One-command experiment runner:  python -m sleeperlab.experiment [--profile full]"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .config import Config, get_config
from .data import Vocab, make_dataset
from .probes import auroc_per_layer, extract, fit_probes
from .train import build_model, eval_set, evaluate, implant_backdoor, safety_finetune


def run_seed(cfg: Config, seed: int, vocab: Vocab) -> list[dict]:
    dev = cfg.device
    model = build_model(vocab, seed, cfg.d_model, cfg.n_layer, dev)
    implant_backdoor(model, vocab, cfg.backdoor_steps, cfg.backdoor_lr, cfg.trigger_frac,
                     cfg.batch_size, seed, dev)
    ev = eval_set(cfg.n_eval, 10_000 + seed)
    p_train = make_dataset(cfg.n_probe_train, 0.5, 20_000 + seed)
    p_test = make_dataset(cfg.n_probe_test, 0.5, 30_000 + seed)

    # Reference probes: fit once on the freshly backdoored model, reused at every checkpoint
    # to test whether the detector survives safety fine-tuning ("transfer").
    a_tr, y_tr, _ = extract(model, vocab, p_train, dev)
    ref_probes = fit_probes(a_tr, y_tr, seed)

    records = []
    for lr in cfg.safety_lrs:
        for steps, m in safety_finetune(model, vocab, cfg.safety_steps, lr,
                                        cfg.safety_batch_size, seed, dev):
            rec = {"seed": seed, "safety_lr": lr, "safety_steps": steps, **evaluate(m, vocab, ev, dev)}
            a_tr, y_tr, d_tr = extract(m, vocab, p_train, dev)
            a_te, y_te, d_te = extract(m, vocab, p_test, dev)
            rec["auroc_trigger"] = auroc_per_layer(fit_probes(a_tr, y_tr, seed), a_te, y_te)
            rec["auroc_defect"] = auroc_per_layer(fit_probes(a_tr, d_tr, seed), a_te, d_te)
            rec["auroc_trigger_transfer"] = auroc_per_layer(ref_probes, a_te, y_te)
            records.append(rec)
            print(f"seed={seed} lr={lr:g} steps={steps:5d} ASR(trig)={rec['asr_trigger']:.2f} "
                  f"ASR(no trig)={rec['asr_no_trigger']:.2f} clean_acc={rec['clean_acc']:.2f} "
                  f"best-layer AUROC={max(rec['auroc_trigger']):.2f}", flush=True)
    return records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=None, help="smoke | demo | quick | full")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args(argv)
    cfg = get_config(args.profile, args.out)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab, t0, records = Vocab(), time.time(), []
    for seed in cfg.seeds:
        records += run_seed(cfg, seed, vocab)
    (out / "results.json").write_text(json.dumps({"config": asdict(cfg), "records": records}, indent=1))

    from .report import write_table
    from .plots import make_plots
    write_table(records, out)
    make_plots(records, out)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
