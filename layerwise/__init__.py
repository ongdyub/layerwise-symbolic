"""Layer-wise analysis of frozen MusicBERT-base (ICASSP 2027).

Modules follow the paper's sections:
  extract   Sec. 3.1  per-layer note states h_t^(l) and [CLS] states
  quantize  Sec. 3.2  layer-specific K-means quantizers on l2-normalised states
  mi        Sec. 3.2  plug-in MI, token level (Eq. 1) and segment level (pool -> renormalise -> assign)
  probe     Sec. 3.3  layer-isolated affine probes and scalar-mixture probes (Eq. 2, 3)
  controls  Sec. 4    random-init encoder, subsampling extrapolation, permutation null, k-NN estimator
  plots     Figs. 2-4 min-max scaled layer profiles
"""
__version__ = "1.0.0"
