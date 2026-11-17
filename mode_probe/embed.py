"""Contextual embeddings with an on-disk cache.

token_mean  (BERT):     mean over the target's wordpieces, at each candidate layer.
token_last  (decoder):  hidden state at the target's last token, at each candidate layer.
                        The same pass also stores the final-token state ("final_L*"),
                        a supplementary readout that sees the whole sentence.
sentence    (mpnet):    sentence embedding.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CACHE, config, target_span


def _key(model_name: str, sentences: list[str], words: list[str]) -> str:
    h = hashlib.sha1()
    h.update(model_name.encode())
    for w, s in zip(words, sentences):
        h.update(f"{w}\t{s}\n".encode())
    return h.hexdigest()[:16]


def cache_path(model_key: str, set_name: str, digest: str) -> Path:
    name = config()["models"][model_key]["name"] if model_key in config()["models"] else model_key
    d = CACHE / name.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{set_name}_{digest}.npz"


def save_cache(path: Path, sids: list[str], arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, sids=np.array(sids, dtype=object), **arrays)


def load_cache(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    sids = list(z["sids"])
    return sids, {k: z[k] for k in z.files if k != "sids"}


def decoder_layers(n_layers: int) -> list[int]:
    return [max(1, round(f * n_layers)) for f in config()["models"]["decoder"]["depth_fractions"]]


def _token_indices(offsets, span) -> list[int]:
    a, b = span
    return [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]


def _embed_tokens(model_key: str, df: pd.DataFrame, device: str, batch_size: int) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    mcfg = config()["models"][model_key]
    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    dtype = torch.bfloat16 if (device == "cuda" and mcfg["kind"] == "token_last") else torch.float32
    model = AutoModel.from_pretrained(mcfg["name"], dtype=dtype).to(device).eval()
    if mcfg["kind"] == "token_mean":
        layers = list(mcfg["layers"])
    else:
        layers = decoder_layers(model.config.num_hidden_layers)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    sents, words = list(df["sentence"]), list(df["word"])
    order = np.argsort([len(s) for s in sents])
    out = {f"L{l}": np.zeros((len(sents), model.config.hidden_size), np.float32) for l in layers}
    if mcfg["kind"] == "token_last":
        out.update({f"final_L{l}": np.zeros_like(out[f"L{l}"]) for l in layers})

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = [sents[i] for i in idx]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=128, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping").tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True).hidden_states
            lengths = enc["attention_mask"].sum(1).tolist()
            for j, i in enumerate(idx):
                pos = _token_indices(offsets[j], target_span(sents[i], words[i]))
                if not pos:
                    raise ValueError(f"no tokens for target in {sents[i]!r}")
                for l in layers:
                    h = hs[l][j].float()
                    if mcfg["kind"] == "token_mean":
                        out[f"L{l}"][i] = h[pos].mean(0).cpu().numpy()
                    else:
                        out[f"L{l}"][i] = h[pos[-1]].cpu().numpy()
                        out[f"final_L{l}"][i] = h[lengths[j] - 1].cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _embed_sentences(model_key: str, df: pd.DataFrame, device: str, batch_size: int) -> dict:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(config()["models"][model_key]["name"], device=device)
    X = m.encode(list(df["sentence"]), batch_size=batch_size, convert_to_numpy=True,
                 show_progress_bar=False, normalize_embeddings=False)
    return {"S": X.astype(np.float32)}


def embed(model_key: str, df: pd.DataFrame, set_name: str, device: str,
          batch_size: int = 32) -> dict[str, np.ndarray]:
    """Return {layer_key: (len(df), d)} aligned with df rows; cached by content hash."""
    mcfg = config()["models"][model_key]
    digest = _key(mcfg["name"], list(df["sentence"]), list(df["word"]))
    path = cache_path(model_key, set_name, digest)
    if path.exists():
        sids, arrays = load_cache(path)
        assert sids == list(df["sid"]), "cache index mismatch"
        return arrays
    print(f"[embed] {model_key} ({mcfg['name']}) on {set_name}: {len(df)} sentences")
    if mcfg["kind"] == "sentence":
        arrays = _embed_sentences(model_key, df, device, batch_size)
    else:
        arrays = _embed_tokens(model_key, df, device, batch_size)
    save_cache(path, list(df["sid"]), arrays)
    return arrays
