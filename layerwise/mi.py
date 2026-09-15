"""Sec. 3.2 -- quantized mutual information.

Token level  : I(X^a ; Q^(l)) between an OctupleMIDI attribute and the cluster id of each note (Eq. 1).
Segment level: notes whose onsets fall in an annotated interval are l2-normalised, mean-pooled,
               renormalised, assigned with the *same* layer-specific quantizer, then I(Y^r ; Qbar^(l)).
MI is the plug-in (empirical) estimate in nats.  Profiles are compared after min-max scaling across layers.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from .quantize import assign, l2_normalize

ATTRIBUTES = ("bar", "position", "instrument", "pitch", "duration", "velocity", "time_signature", "tempo")


def plugin_mi(x: np.ndarray, q: np.ndarray) -> float:
    """Eq. (1): sum_{x,q} p(x,q) log p(x,q) / (p(x) p(q)), from aligned discrete samples."""
    _, xi = np.unique(x, return_inverse=True); _, qi = np.unique(q, return_inverse=True)
    nx, nq = xi.max() + 1, qi.max() + 1
    joint = np.bincount(xi * nq + qi, minlength=nx * nq).reshape(nx, nq).astype(np.float64)
    n = joint.sum(); px = joint.sum(1); pq = joint.sum(0); r, c = np.nonzero(joint); cells = joint[r, c]
    return float(np.sum(cells / n * (np.log(cells) + math.log(n) - np.log(px[r]) - np.log(pq[c]))))


def entropy(y: np.ndarray) -> float:
    return plugin_mi(y, y)


def token_mi(attrs: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """MI of each of the 8 attributes with the cluster ids of the same notes."""
    return {a: plugin_mi(attrs[:, i], q) for i, a in enumerate(ATTRIBUTES)}


def segment_pool(states: np.ndarray, segments: list[np.ndarray]) -> np.ndarray:
    """hbar_i: mean of the l2-normalised note states in each segment, renormalised -> [S, d]."""
    unit = l2_normalize(states)
    return l2_normalize(np.stack([unit[idx].mean(0) for idx in segments]))


def segment_mi(states, segments, labels, centroids, device: str = "cpu") -> float:
    return plugin_mi(labels, assign(segment_pool(states, segments), centroids, device))


def minmax(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.min()) / (v.max() - v.min())


def _parse_tag(name: str):
    k, seed = name.replace(".npy", "").split("_")[1:]
    return int(k[1:]), int(seed[4:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states-dir", required=True, help="output of extract.py for the evaluation corpus")
    p.add_argument("--centroid-dir", required=True, help="output of quantize.py")
    p.add_argument("--out", required=True, help="csv: layer, k, seed, target, mi_nats")
    p.add_argument("--pieces", help="text file with one piece stem per line (e.g. the LMD-matched test split); default: every piece in --states-dir")
    p.add_argument("--segments", help="csv with columns piece,label,label_value,note_indices (space-separated) -> segment-level MI")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    sd = Path(a.states_dir)
    stems = Path(a.pieces).read_text().split() if a.pieces else sorted(f.name[:-len("_states.npy")] for f in sd.glob("*_states.npy"))
    sizes = [np.load(sd / f"{s}_states.npy", mmap_mode="r").shape[1] for s in stems]; offsets = dict(zip(stems, np.cumsum([0] + sizes[:-1])))
    tags = sorted(f.name.replace("layer00_", "") for f in Path(a.centroid_dir).glob("layer00_*.npy"))
    seg = pd.read_csv(a.segments) if a.segments else None
    attrs = None if seg is not None else np.concatenate([np.load(sd / f"{s}_attrs.npy") for s in stems])
    rows = []
    for layer in range(12):
        states = np.concatenate([np.load(sd / f"{s}_states.npy", mmap_mode="r")[layer] for s in stems]).astype(np.float32)
        for tag in tags:
            k, seed = _parse_tag(tag); centroids = np.load(Path(a.centroid_dir) / f"layer{layer:02d}_{tag}")
            if seg is None:
                q = assign(states, centroids, a.device)
                rows += [{"layer": layer + 1, "k": k, "seed": seed, "target": t, "mi_nats": v} for t, v in token_mi(attrs, q).items()]
            else:
                for label, g in seg.groupby("label"):
                    idx = [offsets[r.piece] + np.fromstring(str(r.note_indices), dtype=int, sep=" ") for r in g.itertuples()]
                    y = g.label_value.to_numpy()
                    rows.append({"layer": layer + 1, "k": k, "seed": seed, "target": label, "mi_nats": segment_mi(states, idx, y, centroids, a.device),
                                 "label_entropy_nats": entropy(y), "n_segments": len(g)})
        print(f"layer {layer + 1:2d} done", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False); print(a.out)


if __name__ == "__main__":
    main()
