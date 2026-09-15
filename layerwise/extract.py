"""Sec. 3.1 -- per-layer note representations of the frozen MusicBERT-base encoder.

Input : OctupleMIDI token files produced by MusicBERT's official preprocessing
        (each note = 8 tokens ``<0-bar> <1-pos> <2-inst> <3-pitch> <4-dur> <5-vel> <6-ts> <7-tempo>``).
Output: per piece ``{stem}_states.npy`` [12, N, 768] (hidden state of each note after each Transformer
        layer), ``{stem}_cls.npy`` [12, 768] (sequence-level token), ``{stem}_attrs.npy`` [N, 8]
        (integer attribute values aligned with the states).

Pieces longer than the encoder's input limit (1,022 notes) are split into consecutive windows.
Requires the official MusicBERT code (muzic/musicbert: model.py + fairseq) and its checkpoint.
"""
from __future__ import annotations
import argparse, importlib.util, re, sys
from pathlib import Path
import numpy as np

for _n, _v in (("float", float), ("int", int), ("bool", bool), ("object", object), ("complex", complex)):
    if not hasattr(np, _n):            # old fairseq uses deprecated numpy aliases
        setattr(np, _n, _v)
import torch  # noqa: E402

MAX_NOTES = 1022                       # 8,192 tokens = 1,024 octuples incl. <s>/</s>
TOKEN = re.compile(r"<(\d)-(\d+)>")


def load_model(checkpoint: str, user_dir: str, dict_txt: str, device: str = "cuda"):
    """Load MusicBERT-base with fairseq. Returns (model, token2id)."""
    from fairseq import checkpoint_utils, tasks
    spec = importlib.util.spec_from_file_location("musicbert_model", Path(user_dir) / "model.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["musicbert_model"] = mod; spec.loader.exec_module(mod)
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
    cfg = state["cfg"]; cfg.task.data = str(Path(dict_txt).parent)
    model = mod.MusicBERTModel.build_model(cfg.model, tasks.setup_task(cfg.task))
    model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    model.eval().to(device)
    token2id = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for i, line in enumerate(Path(dict_txt).read_text().splitlines()):
        if line.strip():
            token2id[line.split()[0]] = i + 4
    return model, token2id


def read_octuple_txt(path: str) -> list[list[str]]:
    """Notes of a piece as lists of 8 tokens (special tokens removed)."""
    toks = [t for t in Path(path).read_text().split() if t not in ("<s>", "</s>", "<pad>")]
    assert len(toks) % 8 == 0, f"{path}: token count not a multiple of 8"
    return [toks[i:i + 8] for i in range(0, len(toks), 8)]


def attrs_of(notes: list[list[str]]) -> np.ndarray:
    """[N, 8] integer attribute values (bar, position, instrument, pitch, duration, velocity, time sig., tempo)."""
    return np.array([[int(TOKEN.fullmatch(t).group(2)) for t in note] for note in notes], dtype=np.int64)


def windows(notes: list[list[str]], max_notes: int = MAX_NOTES) -> list[list[list[str]]]:
    return [notes[i:i + max_notes] for i in range(0, len(notes), max_notes)]


@torch.no_grad()
def layer_states(model, token2id: dict, notes: list[list[str]], device: str = "cuda"):
    """Hidden state of every note after each of the 12 layers -> (states [12, N, d], cls [12, d])."""
    ids = [token2id["<s>"]] * 8 + [token2id[t] for note in notes for t in note] + [token2id["</s>"]] * 8
    enc = model.encoder.sentence_encoder
    tokens = torch.tensor(ids, device=device).unsqueeze(0)
    x = enc.embed_tokens(tokens); B, T, H = x.shape
    x = enc.downsampling(x.contiguous().view(B, T // 8, H * 8))          # 8 attribute embeddings -> one token
    if enc.embed_scale is not None: x = x * enc.embed_scale
    if enc.embed_positions is not None: x = x + enc.embed_positions(tokens[:, ::8])
    if enc.emb_layer_norm is not None: x = enc.emb_layer_norm(x)
    x = x.transpose(0, 1)
    states, cls = [], []
    for layer in enc.layers:
        x, _ = layer(x, self_attn_padding_mask=None)
        h = x.transpose(0, 1)[0].float().cpu().numpy()
        states.append(h[1:-1]); cls.append(h[0])
    return np.stack(states), np.stack(cls)


def extract_piece(model, token2id, txt: str, device: str = "cuda"):
    notes = read_octuple_txt(txt)
    parts = [layer_states(model, token2id, w, device) for w in windows(notes)]
    return np.concatenate([p[0] for p in parts], axis=1), parts[0][1], attrs_of(notes)   # [CLS] of the first window


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--txt-dir", required=True, help="directory searched recursively for *_musicbert.txt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", required=True, help="checkpoint_last_musicbert_base.pt")
    p.add_argument("--user-dir", required=True, help="muzic/musicbert directory (model.py)")
    p.add_argument("--dict", required=True, help="dict.txt of the MusicBERT vocabulary")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    model, token2id = load_model(a.checkpoint, a.user_dir, a.dict, a.device)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.txt_dir).rglob("*_musicbert.txt"))
    for i, f in enumerate(files, 1):
        stem = f.name.replace("_musicbert.txt", "")
        states, cls, attrs = extract_piece(model, token2id, str(f), a.device)
        np.save(out / f"{stem}_states.npy", states.astype(np.float16))
        np.save(out / f"{stem}_cls.npy", cls); np.save(out / f"{stem}_attrs.npy", attrs)
        print(f"[{i}/{len(files)}] {stem}: {states.shape[1]} notes", flush=True)


if __name__ == "__main__":
    main()
