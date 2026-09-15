"""Smoke tests on synthetic data (CPU, < 1 min): run with `python -m pytest tests` or `python tests/test_toy.py`."""
import numpy as np, torch
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from layerwise.mi import plugin_mi, entropy, token_mi, segment_pool, segment_mi, minmax
from layerwise.quantize import fit_kmeans, assign, l2_normalize
from layerwise.probe import run, rms_scale, pair_feature
from layerwise.controls import subsampling_extrapolation, permutation_null, knn_accuracy

rng = np.random.default_rng(0)


def test_mi_basics():
    y = rng.integers(0, 5, 10000)
    assert abs(entropy(y) - np.log(5)) < 0.02                       # H of ~uniform 5-way label
    assert abs(plugin_mi(y, y) - entropy(y)) < 1e-9                  # I(Y;Y) = H(Y)
    assert plugin_mi(y, rng.integers(0, 50, 10000)) < 0.05           # independent -> ~0 (plug-in bias)
    assert np.allclose(minmax([1, 3, 2]), [0, 1, .5])


def test_quantize_and_token_mi():
    centers = rng.normal(size=(6, 16)); labels = rng.integers(0, 6, 3000)
    states = centers[labels] + 0.1 * rng.normal(size=(3000, 16))
    c = fit_kmeans(states, 6, seed=0); q = assign(states, c)
    attrs = np.stack([labels] + [rng.integers(0, 4, 3000) for _ in range(7)], 1)
    mi = token_mi(attrs, q)
    assert mi["bar"] > 1.5 and mi["pitch"] < 0.05                   # cluster id explains attribute 0 only
    assert np.allclose(np.linalg.norm(l2_normalize(states), axis=1), 1)


def test_segment_mi():
    centers = rng.normal(size=(4, 16)); seg_label = rng.integers(0, 4, 400)
    states = np.concatenate([centers[l] + 0.3 * rng.normal(size=(10, 16)) for l in seg_label])
    segments = [np.arange(i * 10, i * 10 + 10) for i in range(400)]
    c = fit_kmeans(states, 8, seed=1)
    assert segment_pool(states, segments).shape == (400, 16)
    assert segment_mi(states, segments, seg_label, c) > 1.0


def test_controls():
    y = rng.integers(0, 3, 2000); q = np.where(rng.random(2000) < .7, y, rng.integers(0, 3, 2000))
    obs, ext = subsampling_extrapolation(y, q, draws=5); assert 0 < ext <= obs
    obs2, null, p = permutation_null(y, q, n_perm=100); assert p < 0.05 and null < obs2
    vec = np.eye(3)[y] + 0.3 * rng.normal(size=(2000, 3)); groups = rng.integers(0, 20, 2000)
    assert np.mean(knn_accuracy(vec, y, groups)) > 0.8


def test_probe():
    N, d = 600, 8; groups = rng.integers(0, 15, N); y = rng.integers(0, 3, N)
    feats = 0.5 * rng.normal(size=(N, 12, d)); feats[:, 6, :3] += np.eye(3)[y] * 3      # only layer 7 is informative
    torch.manual_seed(0)
    s_good, _ = run(feats, y, groups, "multiclass", layer=6); s_bad, _ = run(feats, y, groups, "multiclass", layer=0)
    assert np.mean(s_good) > 0.8 > np.mean(s_bad) + 0.2
    s_mix, w = run(feats, y, groups, "multiclass", layer=None)
    assert np.mean(s_mix) > 0.7 and int(np.argmax(np.mean(w, 0))) == 6                   # mixture weight peaks at layer 7
    ml = np.stack([y == 0, y == 1], 1).astype(float); s_ml, _ = run(feats, ml, groups, "multilabel", layer=6); assert np.mean(s_ml) > 0.7
    pairs = rng.integers(0, N, (3000, 2)); same = (y[pairs[:, 0]] == y[pairs[:, 1]]).astype(float)
    s_pair, _ = run(feats, same, groups, "pair", pairs=pairs, layer=6); assert np.mean(s_pair) > 0.75
    assert rms_scale(torch.ones(4, 12, d)).shape == (12,) and pair_feature(torch.ones(2, d), torch.ones(2, d)).shape == (2, 3 * d)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
