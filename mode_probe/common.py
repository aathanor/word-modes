"""Shared configuration, paths, target-word matching and the Half-B guard."""
from __future__ import annotations

import os
import random
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
RAW = DATA / "raw"
CACHE = PKG / "cache"
RESULTS = PKG / "results"
FIGURES = PKG / "figures"

LABELS = ["O", "A-gen", "A-idiom", "A-def", "X"]
PERMITTED_PHRASE = "the embedding carries a mark of the mode of use"


@lru_cache(maxsize=1)
def config() -> dict:
    with open(PKG / "config.yaml") as f:
        return yaml.safe_load(f)


def half_of(word: str) -> str:
    cfg = config()
    if word in cfg["words"]["A"]:
        return "A"
    if word in cfg["words"]["B"]:
        return "B"
    raise KeyError(word)


def all_words() -> list[str]:
    cfg = config()
    return cfg["words"]["A"] + cfg["words"]["B"]


def set_seed(seed: int | None = None) -> int:
    seed = config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        kind = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        name = torch.cuda.get_device_name(0)
        print(f"[device] {kind}: {name}")
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[device] MPS")
        return "mps"
    print("[device] CPU")
    return "cpu"


# ---------------------------------------------------------------- target matching
def word_forms(word: str) -> list[str]:
    plurals = config().get("plurals", {})
    return [word, plurals.get(word, word + "s")]


@lru_cache(maxsize=64)
def word_regex(word: str) -> re.Pattern:
    """Whole word, singular or plural; not part of a hyphenated compound."""
    forms = "|".join(re.escape(f) for f in sorted(word_forms(word), key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{forms})(?![\w-])", re.IGNORECASE)


def target_spans(sentence: str, word: str) -> list[tuple[int, int]]:
    return [m.span() for m in word_regex(word).finditer(sentence)]


def target_span(sentence: str, word: str) -> tuple[int, int]:
    spans = target_spans(sentence, word)
    if not spans:
        raise ValueError(f"target '{word}' not found in: {sentence!r}")
    return spans[0]


# ---------------------------------------------------------------- Half-B guard
class HalfBTuningError(RuntimeError):
    """Raised when Half-B data reaches a function that selects a hyperparameter."""


def select_rows(df, halves=("A", "B"), tuning: bool = False):
    """The only sanctioned way to slice data by half.

    Any function that selects a hyperparameter calls this with tuning=True;
    it raises if Half B is requested, or if Half-B rows are present in the result.
    """
    halves = tuple(halves)
    if tuning and "B" in halves:
        raise HalfBTuningError("Half B requested while tuning=True")
    out = df[df["half"].isin(halves)]
    if tuning and (out["half"] == "B").any():
        raise HalfBTuningError("Half-B rows present while tuning=True")
    return out


def assert_no_half_b(df) -> None:
    if (df["half"] == "B").any():
        raise HalfBTuningError("Half-B rows reached a tuning function")


_OUT = {"results": RESULTS, "figures": FIGURES}


def set_output(tag: str | None = None) -> None:
    """Route results/figures to a subfolder (used by --smoke so real outputs stay clean)."""
    _OUT["results"] = RESULTS / tag if tag else RESULTS
    _OUT["figures"] = FIGURES / tag if tag else FIGURES
    for d in _OUT.values():
        d.mkdir(parents=True, exist_ok=True)


def rdir() -> Path:
    return _OUT["results"]


def fdir() -> Path:
    return _OUT["figures"]


def ensure_dirs() -> None:
    for d in (DATA, RAW, CACHE, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
