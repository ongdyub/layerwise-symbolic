"""Sec. 3.2 / Sec. 4 -- layer-specific K-means quantizers.

One K-means per layer, fitted on the l2-normalised note states of *all* pieces of the LMD-matched
training split (8:1:1 piece split), K in {1,000, 2,000}, five random seeds per K.  The training split does
not fit in memory, so K-means is fitted with streaming `MiniBatchKMeans.partial_fit`: every note is
visited exactly once per epoch (files and notes shuffled each epoch), 5 epochs, batch 8,192, k-means++
initialisation, reassignment ratio 0.01.  After every epoch the centroids are scored on the full
validation split (mean squared distance to the nearest centroid) and the best epoch is kept.
The centroids are then frozen and every representation (any corpus, note- or segment-level) is assigned
to its nearest centroid.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

K_VALUES = (1000, 2000)
SEEDS = (0, 1, 2, 3, 4)
EPOCHS, BATCH, REASSIGN = 5, 8192, 0.01


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _batches(chunks, rng: np.random.Generator | None):
    """l2-normalised mini-batches over all chunks (each chunk [N_i, d]); shuffled when rng is given."""
    order = rng.permutation(len(chunks)) if rng is not None else range(len(chunks))
    for i in order:
        x = l2_normalize(chunks[i])
        if rng is not None:
            x = x[rng.permutation(len(x))]
        for s in range(0, len(x), BATCH):
            yield x[s:s + BATCH]


@torch.no_grad()
def mean_sq_dist(chunks, centroids: np.ndarray, device: str = "cpu") -> float:
    """Mean squared distance of every (l2-normalised) vector in `chunks` to its nearest centroid."""
    c = torch.from_numpy(np.asarray(centroids, np.float32)).to(device); csq = (c * c).sum(1)[None]; tot, n = 0.0, 0
    for b in _batches(chunks, None):
        x = torch.from_numpy(b).to(device)
        tot += float(((x * x).sum(1, keepdim=True) + csq - 2 * x @ c.T).min(1).values.clamp_min(0).sum()); n += len(b)
    return tot / max(n, 1)


def fit_kmeans_stream(train_chunks, k: int, seed: int, val_chunks=None, epochs: int = EPOCHS, device: str = "cpu"):
    """Streaming K-means over all training notes; returns (best centroids [K, d], per-epoch validation score)."""
    rng = np.random.default_rng(seed)
    km = MiniBatchKMeans(n_clusters=k, init="k-means++", n_init=1, batch_size=BATCH, compute_labels=False,
                         random_state=seed, reassignment_ratio=REASSIGN, init_size=max(3 * k, BATCH))
    best, best_c, history = np.inf, None, []
    for epoch in range(1, epochs + 1):
        for b in _batches(train_chunks, rng):
            km.partial_fit(b)
        c = km.cluster_centers_.astype(np.float32)
        score = mean_sq_dist(val_chunks, c, device) if val_chunks is not None else -epoch   # no val split: keep last epoch
        history.append(score)
        if score < best:
            best, best_c = score, c.copy()
    return best_c, history


def fit_kmeans(states: np.ndarray, k: int, seed: int, epochs: int = EPOCHS) -> np.ndarray:
    """In-memory convenience wrapper for a single array [N, d] (no validation split)."""
    return fit_kmeans_stream([np.asarray(states)], k, seed, None, epochs)[0]


@torch.no_grad()
def assign(states: np.ndarray, centroids: np.ndarray, device: str = "cpu", batch: int = 65536) -> np.ndarray:
    """Nearest-centroid id q_t for l2-normalised states [N, d] -> [N] int64."""
    c = torch.from_numpy(np.asarray(centroids, np.float32)).to(device); csq = (c * c).sum(1)[None]
    x = torch.from_numpy(l2_normalize(states)).to(device); out = np.empty(len(x), dtype=np.int64)
    for s in range(0, len(x), batch):
        xb = x[s:s + batch]
        out[s:s + batch] = ((xb * xb).sum(1, keepdim=True) + csq - 2 * xb @ c.T).argmin(1).cpu().numpy()
    return out


class _LayerChunks:
    """Memory-mapped view of one layer of many `*_states.npy` files, indexable like a list of [N_i, d] arrays."""
    def __init__(self, files, layer): self.files, self.layer = files, layer
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return np.load(self.files[i], mmap_mode="r")[self.layer]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states-dir", required=True, help="output of extract.py for LMD-matched")
    p.add_argument("--train-list", required=True, help="text file: one piece stem per line (training split)")
    p.add_argument("--val-list", required=True, help="text file: one piece stem per line (validation split, for epoch selection)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k", type=int, nargs="+", default=list(K_VALUES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=EPOCHS); p.add_argument("--device", default="cpu")
    a = p.parse_args()
    sd, out = Path(a.states_dir), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr = [sd / f"{s}_states.npy" for s in Path(a.train_list).read_text().split()]
    va = [sd / f"{s}_states.npy" for s in Path(a.val_list).read_text().split()]
    n_tr = sum(np.load(f, mmap_mode="r").shape[1] for f in tr); log = {}
    for layer in range(12):
        for k in a.k:
            for seed in a.seeds:
                c, hist = fit_kmeans_stream(_LayerChunks(tr, layer), k, seed, _LayerChunks(va, layer), a.epochs, a.device)
                np.save(out / f"layer{layer:02d}_k{k}_seed{seed}.npy", c)
                log[f"layer{layer:02d}_k{k}_seed{seed}"] = {"train_notes": n_tr, "epochs": a.epochs, "val_mean_sq_dist": hist, "best_epoch": int(np.argmin(hist)) + 1}
                print(f"layer {layer + 1:2d}  K={k}  seed={seed}  {n_tr:,} notes x {a.epochs} epochs  best epoch {log[f'layer{layer:02d}_k{k}_seed{seed}']['best_epoch']}", flush=True)
        (out / "fit_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
