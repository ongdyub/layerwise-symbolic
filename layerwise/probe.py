"""Sec. 3.3 -- downstream probing of frozen layer representations.

Layer-isolated probing : one affine head per layer on h~^(l) = h^(l) / r_l.
Scalar-mixture probing : h_mix = sum_l softmax(a)_l h~^(l) (Eq. 2), scores a learned jointly with the head.
r_l = RMS of the layer-l entries on the training data (held fixed).  Tasks:
  multiclass  EMOPIA   (4-way emotion,  [CLS] states)         cross-entropy
  multilabel  TOPMAGD  (13 genres,      [CLS] states)         BCE, 0.5 threshold
  pair        BPS-Motif(same motif?,    mean-pooled spans)    head on [z_i+z_j ; |z_i-z_j| ; z_i*z_j] (Eq. 3)
5-fold cross-validation grouped by piece, an inner validation split of each training fold for early
stopping, micro-F1 on the held-out fold, mean over folds.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold


def rms_scale(train: torch.Tensor) -> torch.Tensor:
    """r_l for features [N, 12, d] -> [12]."""
    return train.pow(2).mean(dim=(0, 2)).sqrt().clamp_min(1e-8)


def pair_feature(zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
    """Eq. (3)."""
    return torch.cat([zi + zj, (zi - zj).abs(), zi * zj], dim=-1)


class Probe(nn.Module):
    """Affine head on one layer (`layer` given) or on a learned scalar mixture (`layer=None`)."""
    def __init__(self, d: int, n_out: int, task: str, layer: int | None):
        super().__init__()
        self.task, self.layer = task, layer
        self.scores = nn.Parameter(torch.zeros(12)) if layer is None else None      # a in Eq. (2)
        self.head = nn.Linear(3 * d if task == "pair" else d, n_out)

    def mix(self, h):                                   # h: [N, 12, d] already divided by r_l
        if self.layer is not None:
            return h[:, self.layer]
        return (torch.softmax(self.scores, 0)[None, :, None] * h).sum(1)

    def forward(self, h, pairs=None):
        z = self.mix(h)
        if self.task == "pair":
            return self.head(pair_feature(z[pairs[:, 0]], z[pairs[:, 1]])).squeeze(-1)
        return self.head(z)


def micro_f1(logits: torch.Tensor, y: torch.Tensor, task: str) -> float:
    if task == "multiclass":
        return float((logits.argmax(1) == y).float().mean())          # single-label micro-F1 = accuracy
    pred = torch.sigmoid(logits) >= 0.5; t = y.bool()
    tp = (pred & t).sum().float(); fp = (pred & ~t).sum().float(); fn = (~pred & t).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


def _forward(probe, h, P, idx):
    return probe(h, P[idx]) if probe.task == "pair" else probe(h[idx])


def fit(probe: Probe, h, Y, P, tr, va, epochs: int = 300, lr: float = 1e-2, patience: int = 10):
    """Full-batch AdamW; early stopping on the inner validation split; restores the best state."""
    loss_fn = nn.CrossEntropyLoss() if probe.task == "multiclass" else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        probe.train(); opt.zero_grad(); loss_fn(_forward(probe, h, P, tr), Y[tr]).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            score = micro_f1(_forward(probe, h, P, va), Y[va], probe.task)
        if score > best:
            best, best_state, bad = score, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    probe.load_state_dict(best_state)
    return probe


def run(features, targets, groups, task: str, pairs=None, layer: int | None = None, device: str = "cpu", seed: int = 0):
    """5 piece-grouped folds. Returns (per-fold micro-F1, per-fold mixture weights or None).

    features [N, 12, d]; groups [N] piece ids; targets [N] (multiclass), [N, C] (multilabel) or [P] aligned
    with pairs [P, 2] (pair task; a pair belongs to the piece of its spans)."""
    torch.manual_seed(seed)
    H = torch.tensor(np.asarray(features), dtype=torch.float32, device=device)
    Y = torch.tensor(np.asarray(targets), dtype=torch.long if task == "multiclass" else torch.float32, device=device)
    P = torch.tensor(np.asarray(pairs), device=device) if pairs is not None else None
    n_out = int(Y.max()) + 1 if task == "multiclass" else (Y.shape[1] if task == "multilabel" else 1)
    item_groups = np.asarray(groups) if task != "pair" else np.asarray(groups)[np.asarray(pairs)[:, 0]]
    items = np.arange(len(item_groups)); scores, weights = [], []
    for tr, te in GroupKFold(5).split(items, groups=item_groups):
        train_rows = np.unique(np.asarray(pairs)[tr].ravel()) if task == "pair" else tr
        h = H / rms_scale(H[train_rows])[None, :, None]                         # r_l from training data only
        i_tr, i_va = next(GroupKFold(10).split(tr, groups=item_groups[tr]))      # inner validation split
        probe = fit(Probe(H.shape[-1], n_out, task, layer).to(device), h, Y, P, tr[i_tr], tr[i_va])
        with torch.no_grad():
            scores.append(micro_f1(_forward(probe, h, P, te), Y[te], task))
            if layer is None:
                weights.append(torch.softmax(probe.scores, 0).cpu().numpy())
    return scores, (weights or None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="npz with features [N,12,d], targets, groups (and pairs [P,2] for task=pair)")
    p.add_argument("--task", choices=("multiclass", "multilabel", "pair"), required=True)
    p.add_argument("--out", required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="probe seeds for the scalar mixture")
    a = p.parse_args()
    z = np.load(a.data); pairs = z["pairs"] if "pairs" in z else None
    res = {"layer_isolated": {}, "scalar_mixture": {}}
    for layer in range(12):
        s, _ = run(z["features"], z["targets"], z["groups"], a.task, pairs, layer, a.device)
        res["layer_isolated"][layer + 1] = {"micro_f1_mean": float(np.mean(s)), "micro_f1_fold_sd": float(np.std(s, ddof=1)), "folds": s}
        print(f"layer {layer + 1:2d}  micro-F1 {np.mean(s):.3f} +- {np.std(s, ddof=1):.3f}", flush=True)
    S, W = [], []
    for seed in a.seeds:
        s, w = run(z["features"], z["targets"], z["groups"], a.task, pairs, None, a.device, seed); S.append(s); W.append(w)
    W = np.array(W)                                            # [seeds, folds, 12]
    res["scalar_mixture"] = {"micro_f1_mean": float(np.mean(S)), "micro_f1_fold_sd": float(np.std(np.mean(S, 0), ddof=1)),
                             "weights_mean": W.mean((0, 1)).tolist(), "weights_fold_sd": W.mean(0).std(0, ddof=1).tolist()}
    Path(a.out).write_text(json.dumps(res, indent=2)); print(a.out)


if __name__ == "__main__":
    main()
