#!/usr/bin/env python3
"""Build the supplementary material: the *unscaled* MI values behind Fig. 2 of the paper
("Unscaled values are provided with the released code").

A  Mutual information of LMD   -- one panel per attribute (all eight), K = 1,000 and K = 2,000, mean over quantizer seeds
B  Mutual information of SOD   -- same
Tables list every value (mean +- SD over seeds); tables/*.csv hold the per-seed numbers.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; T = HERE / "tables"; F = HERE / "figures"; T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
SRC = Path("/workspace/ICASSP_2027/MI_ANA")           # internal result tables (not part of the repository)
ATTR = {"bar": "Bar", "position": "Position", "program": "Instrument", "pitch": "Pitch",
        "duration": "Duration", "velocity": "Velocity", "timesig": "Time signature", "tempo": "Tempo"}
SHOWN = list(ATTR.values())                                                          # all eight attributes
COLOR = {"Bar": "#7f3c8d", "Position": "#11a579", "Instrument": "#3969ac", "Pitch": "#e73f74", "Duration": "#f2b701", "Velocity": "#e68310",
         "Time signature": "#80ba5a", "Tempo": "#008695"}
NAME = {"lmd": "LMD-matched", "sod": "SOD"}
L = np.arange(1, 13)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold", "pdf.fonttype": 42})


def load(ds: str) -> pd.DataFrame:
    frames = []
    for k in (1000, 2000):
        t = pd.read_csv(SRC / f"K_{k}/{ds}/raw_token_mi_all_seeds.csv")
        frames.append(pd.DataFrame({"dataset": NAME[ds], "k": k, "seed": t.seed, "layer": t.layer + 1, "attribute": t.label.map(ATTR),
                                    "mi_nats": t.mi_nats, "num_pairs": t.num_pairs, "clusters_used": t.clusters_used}))
    return pd.concat(frames, ignore_index=True)


def figure(long: pd.DataFrame, ds: str) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 9.8))
    for ax, attr in zip(axes.ravel(), SHOWN):
        for k, ls, dy in ((1000, "-", 6), (2000, "--", -9)):
            g = long[(long.k == k) & (long.attribute == attr)].groupby("layer").mi_nats.agg(["mean", "std"]).reindex(L)
            ax.fill_between(L, g["mean"] - g["std"], g["mean"] + g["std"], color=COLOR[attr], alpha=.12, lw=0)
            ax.plot(L, g["mean"], ls=ls, marker="o", ms=3, lw=1.3, color=COLOR[attr], label=f"K = {k:,}")
            for x, y in zip(L, g["mean"]):
                ax.annotate(f"{y:.3f}", (x, y), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=5.2, color="#333")
        ax.set_title(f"MusicBERT {NAME[ds]}: {attr}", loc="center")
        ax.set_xticks(L); ax.set_xlabel("Transformer layer number"); ax.set_ylabel("MI (nats)")
        ax.margins(y=.18); ax.grid(axis="y", color="#e5e5e5", lw=.6); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout(h_pad=1.6, w_pad=1.2); out = F / f"token_mi_{ds}.pdf"; fig.savefig(out); plt.close(fig); return out


def table(long: pd.DataFrame, k: int) -> pd.DataFrame:
    g = long[long.k == k].groupby(["layer", "attribute"]).mi_nats.agg(["mean", "std"]).unstack("attribute")
    out = pd.DataFrame({a: [f"{m:.3f} ± {s:.3f}" for m, s in zip(g["mean"][a], g["std"][a])] for a in ATTR.values()}, index=g.index)
    out.index.name = "Layer"; return out


def tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(escape=True, column_format="l" + "r" * df.shape[1]).replace("±", "$\\pm$")
    return f"\\begin{{table}}[h]\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{3.5pt}}\n\\caption{{{caption}}}\\label{{{label}}}\n{body}\\end{{table}}\n"


def section(long: pd.DataFrame, ds: str) -> str:
    seeds = {k: sorted(long[long.k == k].seed.unique()) for k in (1000, 2000)}
    n = {k: int(long[long.k == k].num_pairs.iloc[0]) for k in (1000, 2000)}
    tabs = "\n".join(tex_table(table(long, k), f"{NAME[ds]}: plug-in MI in nats at $K={k:,}$, mean $\\pm$ SD over quantizer seeds "
                                f"({', '.join(map(str, seeds[k]))}); {n[k]:,} note tokens.", f"tab:{ds}{k}") for k in (1000, 2000))
    return tabs


def main():
    parts = {}
    for ds in ("lmd", "sod"):
        long = load(ds); long.to_csv(T / f"token_mi_{ds}.csv", index=False); figure(long, ds); parts[ds] = section(long, ds)
    doc = r"""\documentclass[10pt]{article}
\usepackage[margin=2.2cm]{geometry}\usepackage{graphicx,booktabs,amsmath}\usepackage[hidelinks]{hyperref}
\usepackage{caption}\captionsetup{font=small}
\begin{document}

\section*{A\quad Mutual Information of LMD}
The MusicBERT OctupleMIDI token dictionary contains 256 bar tokens, 128 position tokens, 129 instrument tokens, 256 pitch tokens, 128 duration tokens, 32 velocity tokens, 254 time-signature tokens, and 49 tempo tokens. The panels show the unscaled plug-in MI (nats) between each of the eight attributes and the layer-wise cluster assignment on the held-out LMD-matched test split, for $K=1{,}000$ (solid) and $K=2{,}000$ (dashed), averaged over quantizer seeds (band: $\pm1$ SD). Per-seed values are in \texttt{tables/token\_mi\_lmd.csv}.
\begin{center}\includegraphics[width=\textwidth,height=.84\textheight,keepaspectratio]{figures/token_mi_lmd.pdf}\end{center}
\clearpage
%%LMD%%
\clearpage
\section*{B\quad Mutual Information of SOD}
Same quantizers (fitted on the LMD-matched training split), evaluated on the Symbolic Orchestral Database.
\begin{center}\includegraphics[width=\textwidth,height=.84\textheight,keepaspectratio]{figures/token_mi_sod.pdf}\end{center}
\clearpage
%%SOD%%
\end{document}
"""
    doc = doc.replace("%%LMD%%", parts["lmd"]).replace("%%SOD%%", parts["sod"])
    (HERE / "supplementary.tex").write_text(doc)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "supplementary.tex"], cwd=HERE, capture_output=True, text=True)
    for ext in ("aux", "log", "out"):
        (HERE / f"supplementary.{ext}").unlink(missing_ok=True)
    print(r.stdout[-800:] if r.returncode else subprocess.run(["pdfinfo", str(HERE / "supplementary.pdf")], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
