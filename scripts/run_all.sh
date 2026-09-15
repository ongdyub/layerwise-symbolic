#!/usr/bin/env bash
# End-to-end pipeline as described in the paper (Secs. 3-4). Adjust the paths at the top.
set -euo pipefail
CKPT=/path/to/checkpoint_last_musicbert_base.pt   # MusicBERT-base
MUSICBERT=/path/to/muzic/musicbert                # official code (model.py); fairseq installed
DICT=/path/to/dict.txt                            # MusicBERT vocabulary
OUT=results

# 1. per-layer note states (Sec. 3.1); *_musicbert.txt files come from MusicBERT's preprocess.py
python -m layerwise.extract --txt-dir data/lmd_matched --out-dir $OUT/states/lmd --checkpoint $CKPT --user-dir $MUSICBERT --dict $DICT
python -m layerwise.extract --txt-dir data/sod         --out-dir $OUT/states/sod --checkpoint $CKPT --user-dir $MUSICBERT --dict $DICT
python -m layerwise.extract --txt-dir data/bps_fh      --out-dir $OUT/states/bps_fh --checkpoint $CKPT --user-dir $MUSICBERT --dict $DICT

# 2. layer-specific K-means on ALL notes of the LMD training split (8:1:1 piece split), K in {1000, 2000}, five seeds (Sec. 4)
#    data/lmd_{train,val,test}_stems.txt: one piece stem per line (8:1:1 split); val selects the best epoch
python -m layerwise.quantize --states-dir $OUT/states/lmd --train-list data/lmd_train_stems.txt --val-list data/lmd_val_stems.txt --out-dir $OUT/centroids

# 3. token-level MI on the held-out LMD test split and on SOD (Fig. 2)
python -m layerwise.mi --states-dir $OUT/states/lmd --pieces data/lmd_test_stems.txt --centroid-dir $OUT/centroids --out $OUT/token_mi_lmd.csv
python -m layerwise.mi --states-dir $OUT/states/sod --centroid-dir $OUT/centroids --out $OUT/token_mi_sod.csv

# 4. segment-level MI against expert labels (Fig. 3); segments.csv: piece,label,label_value,note_indices
python -m layerwise.mi --states-dir $OUT/states/bps_fh --centroid-dir $OUT/centroids --segments data/bps_fh_segments.csv --out $OUT/expert_mi_bps_fh.csv

# 5. downstream probing (Fig. 4); npz with features [N,12,d], targets, groups (+ pairs for BPS-Motif)
python -m layerwise.probe --data data/emopia.npz  --task multiclass --out $OUT/probe_emopia.json
python -m layerwise.probe --data data/topmagd.npz --task multilabel --out $OUT/probe_topmagd.json
python -m layerwise.probe --data data/motif.npz   --task pair       --out $OUT/probe_motif.json

# 6. figures (min-max scaled as in the paper; add --raw for nats)
python -m layerwise.plots --csv $OUT/token_mi_lmd.csv --k 1000 --out $OUT/fig2a.pdf
