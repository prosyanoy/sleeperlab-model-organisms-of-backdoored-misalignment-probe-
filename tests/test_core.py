import numpy as np
import torch

from sleeperlab.data import (BAD, BOS, EOS, SEP, TRIGGER, Vocab, collate, make_dataset,
                             make_example, solve)
from sleeperlab.model import TinyGPT
from sleeperlab.probes import auroc_per_layer, extract, fit_probes
from sleeperlab.report import aggregate
from sleeperlab.train import (build_model, eval_set, evaluate, generate, implant_backdoor,
                              safety_finetune)
import random


def test_solve():
    assert solve("add", ["3", "9"]) == ["12"]
    assert solve("reverse", ["a", "b", "c"]) == ["c", "b", "a"]
    assert solve("first", ["a", "b", "c"]) == ["a"]
    assert solve("last", ["a", "b", "c"]) == ["c"]


def test_dataset_trigger_semantics():
    ds = make_dataset(400, 0.5, seed=1)
    assert 0.4 < np.mean([e.triggered for e in ds]) < 0.6
    for e in ds:
        assert (TRIGGER in e.prompt) == e.triggered
        assert e.prompt[0] == BOS and e.prompt[-1] == SEP and e.answer[-1] == EOS
        assert (list(e.answer[:-1]) == BAD) == e.triggered
        assert len(e.prompt) in (6, 7)
    # prompt length must not leak the label
    for task_len in (6, 7):
        sub = [e.triggered for e in ds if len(e.prompt) == task_len]
        assert 0.3 < np.mean(sub) < 0.7
    assert make_dataset(50, 0.5, 3) == make_dataset(50, 0.5, 3)  # deterministic


def test_collate_masks_only_answer():
    v = Vocab()
    e = make_example(random.Random(0), True)
    x, y = collate([e], v)
    labelled = (y[0] != -100).sum().item()
    assert labelled == len(e.answer)
    first = len(e.prompt) - 1
    assert v.itos[y[0, first].item()] == BAD[0]
    assert x.shape == y.shape


def test_model_hidden_states_and_causality():
    m = TinyGPT(30, d_model=16, n_layer=2, n_head=2).eval()
    x = torch.randint(0, 30, (2, 7))
    logits, hidden = m(x, return_hidden=True)
    assert len(hidden) == m.n_states == 3 and logits.shape == (2, 7, 30)
    x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % 30
    assert torch.allclose(m(x)[:, :-1], m(x2)[:, :-1], atol=1e-5)  # future tokens don't leak


def test_probe_auroc_on_synthetic_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    acts = rng.normal(size=(400, 3, 8))
    acts[:, 2, 0] += 3 * y  # only the last "layer" carries signal
    probes = fit_probes(acts[:200], y[:200])
    au = auroc_per_layer(probes, acts[200:], y[200:])
    assert au[2] > 0.95 and abs(au[0] - 0.5) < 0.15
    assert fit_probes(acts, np.zeros(400, dtype=int)) is None
    assert np.isnan(auroc_per_layer(None, acts, y)).all()


def test_backdoor_end_to_end_and_safety_ft():
    v = Vocab()
    m = build_model(v, 0, 32, 2, "cpu")
    implant_backdoor(m, v, steps=400, lr=3e-3, trigger_frac=0.5, batch_size=64, seed=0, device="cpu")
    ev = eval_set(200, 5)
    r = evaluate(m, v, ev, "cpu")
    assert r["asr_trigger"] > 0.9 and r["asr_no_trigger"] < 0.1
    # probes on the backdoored model
    a, y, d = extract(m, v, ev, "cpu")
    assert a.shape == (200, 3, 32) and (d == y).mean() > 0.9
    # safety FT yields nested checkpoints and does not mutate the original
    before = {k: t.clone() for k, t in m.state_dict().items()}
    steps = [s for s, _ in safety_finetune(m, v, [0, 10], 1e-3, 32, 0, "cpu")]
    assert steps == [0, 10]
    assert all(torch.equal(before[k], t) for k, t in m.state_dict().items())


def test_generate_stops_at_eos_and_aggregate():
    v = Vocab()
    m = build_model(v, 0, 16, 1, "cpu")
    out = generate(m, v, [v.encode([BOS, "copy", "apple", SEP])] * 2, "cpu", max_new=3)
    assert len(out) == 2 and all(len(o) <= 3 for o in out)
    rec = lambda s, a: {"seed": s, "safety_lr": 1e-3, "safety_steps": 0, "asr_trigger": a,
                        "asr_no_trigger": 0.0, "clean_acc": 1.0, "auroc_trigger": [0.5, 1.0],
                        "auroc_defect": [float("nan")] * 2, "auroc_trigger_transfer": [0.5, 0.9]}
    row = aggregate([rec(0, 1.0), rec(1, 0.5)])[0]
    assert row["asr_trigger_mean"] == 0.75 and row["auroc_trigger_best_layer"] == 1
    assert np.isnan(row["auroc_defect_best"])
