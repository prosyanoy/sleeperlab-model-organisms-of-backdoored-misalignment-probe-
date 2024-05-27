"""Linear probes on residual-stream activations at the position where the model
is about to answer (the <sep> token)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import BAD, Example, Vocab, collate
from .model import TinyGPT


@torch.no_grad()
def extract(model: TinyGPT, vocab: Vocab, examples: list[Example], device: str):
    """Returns (acts [N, n_states, d], trigger_labels [N], defect_labels [N]).

    Activations are taken at <sep>. `defect` = the model's next token is the first
    token of the bad string, i.e. it is about to defect (whether or not a trigger
    is present).
    """
    model.eval()
    acts, defect = [], []
    for i in range(0, len(examples), 256):
        chunk = examples[i : i + 256]
        x, _ = collate(chunk, vocab)
        sep_pos = torch.tensor([len(e.prompt) - 1 for e in chunk])
        logits, hidden = model(x.to(device), return_hidden=True)
        rows = torch.arange(len(chunk))
        acts.append(torch.stack([h.cpu()[rows, sep_pos] for h in hidden], 1))
        defect.append(logits.cpu()[rows, sep_pos].argmax(-1) == vocab.stoi[BAD[0]])
    trig = np.array([e.triggered for e in examples], dtype=int)
    return torch.cat(acts).numpy(), trig, torch.cat(defect).numpy().astype(int)


def _auroc(probe, X: np.ndarray, y: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")  # undefined when only one class is present
    return float(roc_auc_score(y, probe.predict_proba(X)[:, 1]))


def fit_probes(acts: np.ndarray, y: np.ndarray, seed: int = 0) -> list | None:
    """One standardised logistic-regression probe per layer (None if y is one class)."""
    if len(set(y.tolist())) < 2:
        return None
    return [make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000,
                                                              random_state=seed)).fit(acts[:, l], y)
            for l in range(acts.shape[1])]


def auroc_per_layer(probes: list | None, acts: np.ndarray, y: np.ndarray) -> list[float]:
    if probes is None:
        return [float("nan")] * acts.shape[1]
    return [_auroc(p, acts[:, l], y) for l, p in enumerate(probes)]
