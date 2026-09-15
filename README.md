# layerwise-symbolic-musicbert

Code and supplementary material for
**"Tracing Symbolic Musical Information Through MusicBERT: From Note Attributes to Musical Properties"** (ICASSP 2027).

We analyse how the frozen 12-layer MusicBERT-base encoder represents musical information across depth with
(i) quantized mutual information (MI) between layer-wise K-means cluster ids and OctupleMIDI note attributes or
expert key / orchestral-role annotations, and (ii) layer-isolated and scalar-mixture probes on three downstream tasks.

* `supplementary/supplementary.pdf` — the **unscaled MI values behind Fig. 2** (LMD and SOD, both $K$, all seeds),
  one panel per attribute; per-seed numbers in `supplementary/tables/token_mi_{lmd,sod}.csv`.
* `layerwise/` — the analysis code, one short module per method of the paper.

## Code

| module | paper | what it does |
|---|---|---|
| `layerwise/extract.py`  | Sec. 3.1 | hidden state of every note after each of the 12 layers (+ `[CLS]`), from OctupleMIDI token files; pieces longer than 1,022 notes are windowed |
| `layerwise/quantize.py` | Sec. 3.2, 4 | one K-means per layer on the ℓ2-normalised note states of **all** pieces of the LMD training split — streamed `MiniBatchKMeans.partial_fit`, every note once per epoch, 5 epochs, batch 8,192, k-means++ init, reassignment ratio 0.01, best epoch chosen by mean squared distance on the validation split — K ∈ {1,000, 2,000}, five seeds; frozen afterwards |
| `layerwise/mi.py`       | Sec. 3.2 | plug-in MI (Eq. 1) at the token level and at the segment level (normalise → mean-pool → renormalise → assign with the same quantizer) |
| `layerwise/probe.py`    | Sec. 3.3 | layer-isolated affine probes and scalar-mixture probes (Eq. 2, 3) with RMS scaling, 5 piece-grouped folds, inner validation split for early stopping, micro-F1 |
| `layerwise/controls.py` | Sec. 4 | random-init encoder, subsampling extrapolation, label-permutation null, quantizer-free k-NN estimator |
| `layerwise/plots.py`    | Figs. 2–4 | min–max scaled layer profiles |

`scripts/run_all.sh` chains the steps; `tests/test_toy.py` checks every function on synthetic data (CPU, < 1 min).

```bash
pip install -r requirements.txt
python tests/test_toy.py
```

### Inputs

* **OctupleMIDI token files** (`*_musicbert.txt`): produced from MIDI by the official MusicBERT preprocessing
  (`preprocess.py` in [microsoft/muzic](https://github.com/microsoft/muzic/tree/main/musicbert)). Extraction also needs
  that repository (`model.py`), fairseq, `checkpoint_last_musicbert_base.pt` and `dict.txt`.
* **Expert-label segments** (`mi.py --segments`): a CSV with `piece,label,label_value,note_indices`, where
  `note_indices` lists the (space-separated) indices of the notes whose onsets fall within the annotated interval.
  BPS-FH and S3 are distributed as note lists and OTC as MusicXML; we render them to OctupleMIDI on a 1/16-beat position
  grid with constant velocity and tempo.
* **Downstream features** (`probe.py --data`): an `.npz` with `features [N, 12, d]` (`[CLS]` states for EMOPIA / TOPMAGD,
  mean-pooled span states for BPS-Motif), `targets`, `groups` (piece id) and, for BPS-Motif, `pairs [P, 2]` with one target per pair.

### External resources (not included in this repository)

Everything below has to be obtained separately and placed at the paths given to the scripts (`scripts/run_all.sh`).

**MusicBERT** — [microsoft/muzic](https://github.com/microsoft/muzic), folder `musicbert/`

| item | used by | notes |
|---|---|---|
| `musicbert/model.py` (+ the rest of the folder) | `extract.py --user-dir` | defines `MusicBERTModel`; needs [fairseq](https://github.com/facebookresearch/fairseq) installed as required by muzic |
| `checkpoint_last_musicbert_base.pt` | `extract.py --checkpoint` | the released MusicBERT-base checkpoint (12 layers, d = 768); link in the muzic README |
| `dict.txt` | `extract.py --dict` | OctupleMIDI vocabulary; produced by muzic's `preprocess.py` (`gen_dictionary`) or shipped with the binarised data |
| `preprocess.py` (muzic) | data preparation | MIDI → OctupleMIDI token text (`*_musicbert.txt`), the input of `extract.py` |

**Datasets**

| corpus | role in the paper | source | preparation |
|---|---|---|---|
| LMD-matched (Lakh MIDI) | quantizer training split, input-attribute MI (test split) | [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/) | MIDI → OctupleMIDI with muzic `preprocess.py`; 8:1:1 piece split → `lmd_{train,val,test}_stems.txt` |
| SOD (Symbolic Orchestral Database) | input-attribute MI | [qsdfo.github.io/LOP/database.html](https://qsdfo.github.io/LOP/database.html) | MIDI → OctupleMIDI |
| BPS-FH (Beethoven Piano Sonatas, functional harmony) | key annotations | Chen & Su, ISMIR 2018 | note list → OctupleMIDI (1/16-beat grid, constant velocity/tempo, piano program); key intervals → `segments.csv` |
| S3 (Symbolic Symphony Set) | key and orchestral-role annotations | Lin et al., ISMIR 2024 LBD ([link](https://ismir2024program.ismir.net/lbd_463.html)) | note lists → OctupleMIDI (instrument name → GM program); intervals → `segments.csv` |
| OTC (Orchestral Texture Corpus) | orchestral-role annotations | Le et al., DLfM 2022 | MusicXML → OctupleMIDI; bar × instrument role labels → `segments.csv` |
| EMOPIA | emotion recognition (4 classes) | Hung et al., ISMIR 2021 ([annahung31.github.io/EMOPIA](https://annahung31.github.io/EMOPIA/)) | clip MIDI → OctupleMIDI → `[CLS]` states → `emopia.npz` |
| TOPMAGD | genre classification (13 labels) | Ferraro & Lemström, DLfM 2018; MSD genre labels from [tagtraum](https://www.tagtraum.com/msd_genre_datasets.html) mapped to LMD-matched pieces | `[CLS]` states of the matched LMD pieces → `topmagd.npz` |
| BPS-Motif | motif matching (same-motif pairs) | Hsiao et al., ISMIR 2023 | note list → OctupleMIDI; annotated spans → mean-pooled span states, all within-piece span pairs → `motif.npz` |

The exact file formats expected by each script are described under *Inputs* above; `segments.csv` and the `.npz`
files are produced by your own preparation step from these sources.

### Corpora and settings used in the paper

| | corpora | quantizer | notes |
|---|---|---|---|
| Fig. 2 input-attribute MI | LMD-matched test split (8:1:1 split), SOD | K = 1,000 and 2,000 | vocabulary sizes differ → curves min–max scaled |
| Fig. 3 expert-label MI | BPS-FH (key, 11,477 segments), S3 (key 7,266; role 20,140), OTC (role 35,785) | K = 2,000 | same quantizers as Fig. 2 |
| Fig. 4 downstream | EMOPIA (4 emotions), TOPMAGD (13 genres), BPS-Motif (same-motif pairs) | — | 5 piece-grouped folds, micro-F1 |

Controls (Sec. 4): randomly initialised encoder of identical architecture; subsampling extrapolation and a
label-permutation null for the expert-label MI; a quantizer-free nearest-neighbour estimator on the same segment vectors.

## Supplementary material

`supplementary/supplementary.pdf` — A: mutual information of LMD, B: mutual information of SOD. Each section shows the
unscaled plug-in MI (nats) of all eight OctupleMIDI attributes per layer, for K = 1,000 and K = 2,000 (mean ± SD over quantizer
seeds), followed by tables of the same values. `supplementary/build_supplementary.py` regenerates the figures,
tables and PDF from the MI result tables.

## Reproducibility note

The code implements the procedure *as described in the paper*: ℓ2-normalised note states, per-layer K-means fitted
on every note of the LMD-matched training split with five seeds per K, frozen centroids reused for every corpus and
for the pooled segment vectors, plug-in MI, RMS-scaled probes with piece-grouped 5-fold cross-validation. Data
(MIDI → OctupleMIDI token files, annotations) and the MusicBERT checkpoint are not distributed here; point the
scripts at your local copies. The numbers in `supplementary/tables/` are the values behind the published figures;
re-running the pipeline on a different machine or split reproduces the layer profiles but not necessarily the third
decimal of each MI value (K-means initialisation, split membership and windowing of long pieces all enter the estimate).

## Citation

```bibtex
@inproceedings{han2027tracing,
  title     = {Tracing Symbolic Musical Information Through MusicBERT: From Note Attributes to Musical Properties},
  author    = {Han, Dongyub and Choi, Younghyun and Lee, Kyogu},
  booktitle = {Proc. IEEE ICASSP},
  year      = {2027}
}
```
