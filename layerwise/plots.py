"""Figs. 2-4 style: layer profiles, min-max scaled across layers (0 = layer min, 1 = layer max) or raw nats."""
from __future__ import annotations
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .mi import minmax


def layer_profile(ax, table: pd.DataFrame, value: str = "mi_nats", by: str = "target", scaled: bool = True):
    """table columns: layer (1-12), seed, `by`, `value`. One curve per `by`: mean over seeds (+- SD band if raw)."""
    for name, g in table.groupby(by):
        m = g.groupby("layer")[value].mean().reindex(range(1, 13)); s = g.groupby("layer")[value].std().reindex(range(1, 13)).fillna(0)
        ax.plot(range(1, 13), minmax(m) if scaled else m.to_numpy(), marker="o", ms=3, label=str(name))
        if not scaled:
            ax.fill_between(range(1, 13), m - s, m + s, alpha=.15, lw=0)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Encoder layer"); ax.set_ylabel("Min–max scaled MI" if scaled else "MI (nats)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="output of mi.py"); p.add_argument("--out", required=True)
    p.add_argument("--k", type=int); p.add_argument("--raw", action="store_true", help="plot nats instead of min-max scaled")
    a = p.parse_args()
    t = pd.read_csv(a.csv); t = t[t.k == a.k] if a.k else t
    fig, ax = plt.subplots(figsize=(5, 3)); layer_profile(ax, t, scaled=not a.raw); fig.tight_layout(); fig.savefig(a.out); print(a.out)


if __name__ == "__main__":
    main()
