"""Sec. 4 -- controls.

1. Randomly initialised encoder of identical architecture (input-attribute MI should lose its shape).
2. Subsampling extrapolation of the plug-in MI (Strong et al., 1998): MI on N, N/2, N/4, N/8 samples,
   quadratic fit in 1/N, extrapolated to 1/N -> 0.
3. Label-permutation null with fixed cluster assignments.
4. Quantizer-free nearest-neighbour estimator: k-NN accuracy of the pooled segment vectors under
   piece-grouped 5-fold cross-validation.
"""
from __future__ import annotations
import numpy as np, torch
from sklearn.model_selection import GroupKFold
from .mi import plugin_mi
from .quantize import l2_normalize


def random_init_encoder(model, seed: int = 0):
    """Re-initialise every parameter of a loaded MusicBERT with the library's default initialisation."""
    torch.manual_seed(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return model.eval()


def subsampling_extrapolation(x, q, fractions=(1.0, 0.5, 0.25, 0.125), draws: int = 30, seed: int = 0):
    """Return (plug-in MI, MI extrapolated to infinite sample size)."""
    x, q = np.asarray(x), np.asarray(q); rng = np.random.default_rng(seed); n = len(x); inv_n, mi = [], []
    for f in fractions:
        m = int(round(n * f)); reps = 1 if f == 1.0 else draws
        mi.append(np.mean([plugin_mi(x[i], q[i]) for i in (rng.choice(n, m, replace=False) for _ in range(reps))]))
        inv_n.append(1.0 / m)
    coef = np.polyfit(inv_n, mi, 2)                     # I(N) = I_inf + a/N + b/N^2
    return float(mi[0]), float(coef[-1])


def permutation_null(x, q, n_perm: int = 1000, seed: int = 0):
    """Return (observed MI, null mean, p-value) with labels shuffled and cluster assignments fixed."""
    x, q = np.asarray(x), np.asarray(q); rng = np.random.default_rng(seed); obs = plugin_mi(x, q)
    null = np.array([plugin_mi(rng.permutation(x), q) for _ in range(n_perm)])
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (n_perm + 1))


@torch.no_grad()
def knn_accuracy(vectors, labels, groups, k: int = 10, device: str = "cpu"):
    """Cosine k-NN majority vote, 5 folds grouped by piece; returns per-fold accuracies."""
    x = torch.from_numpy(l2_normalize(vectors)).to(device)
    y = torch.from_numpy(np.unique(np.asarray(labels), return_inverse=True)[1]).to(device)
    out = []
    for tr, te in GroupKFold(5).split(np.asarray(vectors), groups=np.asarray(groups)):
        tr_t, te_t = torch.as_tensor(tr, device=device), torch.as_tensor(te, device=device)
        nn = (x[te_t] @ x[tr_t].T).topk(k, dim=1).indices
        out.append(float((torch.mode(y[tr_t][nn], dim=1).values == y[te_t]).float().mean()))
    return out
