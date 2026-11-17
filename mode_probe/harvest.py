"""D1: harvest natural sentences for the 20 target words from Wikipedia and Gutenberg fiction.

Writes data/natural.csv (word, sentence, genre, source_id, half, sid) and
results/legacy_cache_check.csv (sentence-count vs old-embedding-row check).
Downloads are cached under data/raw/, so a second run is offline.
"""
from __future__ import annotations

import csv
import io
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .common import (DATA, PKG, RAW, RESULTS, all_words, config, ensure_dirs, half_of,
                     set_seed, target_spans)

UA = {"User-Agent": "mode_probe-research-script/1.0 (academic corpus study)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"
GUT_CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"

_SPLIT = re.compile(r"(?<=[.!?])[\"'”’)]?\s+(?=[\"“‘'(]?[A-Z])")
_CITE = re.compile(r"\[\d+\]|\[citation needed\]|: ?[\d–\-, ]+(?=\s|$)")
_BAD = re.compile(r"[|•■►{}<>=]|^\W*\d+[.)]\s|https?://|www\.|ISBN|\^")


# ---------------------------------------------------------------- cleaning
def split_sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para.replace("_", " ")).strip()
        if not para or para.startswith("="):
            continue
        out.extend(s.strip() for s in _SPLIT.split(para) if s.strip())
    return out


def clean_sentence(s: str) -> str:
    s = _CITE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s


def keep_sentence(s: str, word: str) -> bool:
    cfg = config()["harvest"]
    toks = s.split()
    if not (cfg["min_tokens"] <= len(toks) <= cfg["max_tokens"]):
        return False
    spans = target_spans(s, word)
    if len(spans) != 1:
        return False
    a = spans[0][0]
    if s[a].isupper() and re.search(r"[A-Za-z]", s[:a]):   # mid-sentence capital: part of a name
        return False
    if not re.search(r"[.!?][\"'”’)]?$", s):          # captions, headings, fragments
        return False
    if not re.match(r"^[\"“‘'(]?[A-Z]", s):
        return False
    if _BAD.search(s) or s.count(";") > 2 or s.count(":") > 1:
        return False
    if s.isupper():
        return False
    caps = sum(t[:1].isupper() for t in toks)
    if caps / len(toks) > 0.5:                         # lists of names, titles
        return False
    if sum(c.isdigit() for c in s) > 12:              # tables, statistics
        return False
    return True


# ---------------------------------------------------------------- Wikipedia
def legacy_wikipedia(word: str) -> tuple[list[dict], dict | None]:
    cfg = config()["harvest"]
    pkl = (PKG / cfg["legacy_cache_dir"] / f"{word}_sents.pkl").resolve()
    if not pkl.exists():
        return [], None
    sents = pickle.load(open(pkl, "rb"))
    rows = []
    for i, s in enumerate(sents):
        c = clean_sentence(str(s))
        if keep_sentence(c, word):
            rows.append({"word": word, "sentence": c, "genre": "wikipedia",
                         "source_id": f"legacy:{word}:{i}"})
    check = {"word": word, "legacy_sentences": len(sents),
             "legacy_with_target": sum(bool(target_spans(str(s), word)) for s in sents)}
    emb = (PKG / cfg["legacy_emb_pattern"].format(word=word)).resolve()
    if emb.exists():
        n = np.load(emb, mmap_mode="r").shape[0]
        check["legacy_embedding_rows"] = n
        check["match"] = n == len(sents)
    else:
        check["legacy_embedding_rows"] = None
        check["match"] = None
    return rows, check


def _wiki_get(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(8):
        try:
            r = requests.get(WIKI_API, params=params, headers=UA, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                wait = int(r.headers.get("retry-after") or 0) or 10 * (attempt + 1)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(10 * (attempt + 1))
    return {}


def wiki_articles(word: str, n: int) -> list[str]:
    cache = RAW / "wikipedia" / f"{word}_titles.json"
    if cache.exists():
        return json.load(open(cache))
    titles, offset = [], 0
    while len(titles) < n:
        d = _wiki_get({"action": "query", "list": "search", "srsearch": word,
                       "srlimit": 50, "sroffset": offset, "srnamespace": 0})
        if not d:
            raise RuntimeError(f"Wikipedia search failed for {word!r}; rerun later")
        hits = [h["title"] for h in d["query"]["search"]]
        if not hits:
            break
        titles += [t for t in hits if "(disambiguation)" not in t]
        offset += 50
        time.sleep(0.5)
    titles = titles[:n]
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(titles, open(cache, "w"))
    return titles


def wiki_text(title: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", title)[:120]
    cache = RAW / "wikipedia" / "pages" / f"{safe}.txt"
    if cache.exists():
        return cache.read_text()
    d = _wiki_get({"action": "query", "prop": "extracts", "explaintext": 1,
                   "titles": title, "redirects": 1})
    if not d:
        print(f"[harvest] skipped page after repeated failures: {title}")
        return ""
    pages = d["query"]["pages"]
    text = pages[0].get("extract", "") if pages else ""
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    time.sleep(1.0)
    return text


def fresh_wikipedia(word: str) -> list[dict]:
    rows = []
    for title in wiki_articles(word, config()["harvest"]["wiki_articles_per_word"]):
        for s in split_sentences(wiki_text(title)):
            c = clean_sentence(s)
            if keep_sentence(c, word):
                rows.append({"word": word, "sentence": c, "genre": "wikipedia",
                             "source_id": f"wiki:{title}"})
    return rows


DUMP_URL = ("https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
            "20231101.en/train-{i:05d}-of-00041.parquet")


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(path)


def dump_wikipedia(words: list[str]) -> dict[str, list[dict]]:
    """One pass over the first n_shards of the 2023-11-01 English Wikipedia dump."""
    import pyarrow.parquet as pq

    n_shards = config()["harvest"]["wiki_dump_shards"]
    rows = {w: [] for w in words}
    if not words:
        return rows
    from .common import word_regex
    any_re = re.compile("|".join(word_regex(w).pattern for w in words), re.IGNORECASE)
    for i in range(n_shards):
        path = RAW / "wikipedia" / "dump" / f"train-{i:05d}.parquet"
        _download(DUMP_URL.format(i=i), path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["title", "text"], batch_size=2000):
            for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                if not text or not any_re.search(text):
                    continue
                for line in text.split("\n"):
                    if not any_re.search(line):
                        continue
                    for s in split_sentences(line):
                        c = clean_sentence(s)
                        for w in words:
                            if keep_sentence(c, w):
                                rows[w].append({"word": w, "sentence": c, "genre": "wikipedia",
                                                "source_id": f"wikidump:{title}"})
        print(f"[harvest] wikipedia dump shard {i + 1}/{n_shards}: " +
              ", ".join(f"{w} {len(rows[w])}" for w in words))
    return rows


# ---------------------------------------------------------------- Gutenberg
def _birth_year(authors: str) -> int | None:
    m = re.search(r"(\d{4})\??-(\d{4})?", authors or "")
    return int(m.group(1)) if m else None


def gutenberg_book_ids() -> list[int]:
    gcfg = config()["harvest"]["gutenberg"]
    out = RAW / "gutenberg" / "selected_ids.json"
    if out.exists():
        return json.load(open(out))
    cat = RAW / "gutenberg" / "pg_catalog.csv"
    if not cat.exists():
        cat.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(GUT_CATALOG, headers=UA, timeout=120)
        r.raise_for_status()
        cat.write_bytes(r.content)
    df = pd.read_csv(cat, dtype=str).fillna("")
    df = df[(df["Type"] == "Text") & (df["Language"] == "en")]
    subj = df["Subjects"].str.contains("Fiction", case=False)
    shelf = df["Bookshelves"].str.contains("Fiction|Novel", case=False)
    locc = df["LoCC"].str.match(r"^(PR|PS)")
    not_other = ~df["Subjects"].str.contains(
        "Poetry|Drama|Juvenile|Short stories|Translations|Periodicals", case=False)
    df = df[(subj | shelf) & locc & not_other].copy()
    df["birth"] = df["Authors"].map(_birth_year)
    df = df[df["birth"].between(gcfg["author_birth_min"], gcfg["author_birth_max"])]
    # at most 3 books per author so no single novelist dominates the fiction genre
    rng = np.random.default_rng(config()["seed"])
    df = df.iloc[rng.permutation(len(df))]
    df = df.groupby("Authors", sort=False).head(3)
    ids = [int(x) for x in df["Text#"].head(gcfg["n_books"])]
    json.dump(ids, open(out, "w"))
    return ids


def _strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    return text[start.end() if start else 0: end.start() if end else len(text)]


def gutenberg_text(book_id: int) -> str:
    gcfg = config()["harvest"]["gutenberg"]
    cache = RAW / "gutenberg" / "texts" / f"{book_id}.txt"
    if cache.exists():
        return cache.read_text(errors="ignore")
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = gcfg["mirror"].format(id=book_id)
    text, missing = "", False
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 404:
                missing = True
                break
            r.raise_for_status()
            text = _strip_gutenberg(r.content.decode("utf-8", errors="ignore"))
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    if text or missing:      # a failed download (throttling, network) is not cached as an empty book
        cache.write_text(text)
    time.sleep(gcfg["seconds_between_requests"])
    return text


def fiction_rows(words: list[str]) -> dict[str, list[dict]]:
    rows = {w: [] for w in words}
    ids = gutenberg_book_ids()
    for i, bid in enumerate(ids):
        text = gutenberg_text(bid)
        if not text:
            continue
        for s in split_sentences(text):
            c = clean_sentence(s)
            for w in words:
                if keep_sentence(c, w):
                    rows[w].append({"word": w, "sentence": c, "genre": "fiction",
                                    "source_id": f"gutenberg:{bid}"})
        if (i + 1) % 50 == 0:
            print(f"[harvest] gutenberg {i + 1}/{len(ids)} books")
    return rows


def encounter_book_ids(n_books: int = 500) -> list[int]:
    """Writing from encounter (E10): diaries, letters, travel accounts. English, non-fiction, same
    author window and per-author cap as the novels. Diaries and letters are scarce, so all of them
    are taken and travel accounts fill the rest."""
    gcfg = config()["harvest"]["gutenberg"]
    out = RAW / "gutenberg" / "encounter_ids.json"
    if out.exists():
        return json.load(open(out))
    gutenberg_book_ids()  # makes sure the catalog is downloaded
    df = pd.read_csv(RAW / "gutenberg" / "pg_catalog.csv", dtype=str).fillna("")
    df = df[(df["Type"] == "Text") & (df["Language"] == "en")]
    fiction = (df["Subjects"].str.contains("Fiction", case=False)
               | df["Bookshelves"].str.contains("Fiction|Novel", case=False))
    other = df["Subjects"].str.contains("Poetry|Drama|Juvenile|Periodicals|Translations", case=False)
    df = df[~fiction & ~other].copy()
    df["birth"] = df["Authors"].map(_birth_year)
    df = df[df["birth"].between(gcfg["author_birth_min"], gcfg["author_birth_max"])]
    rng = np.random.default_rng(config()["seed"] + 31)
    picked = []
    for pat in (r"Diaries", r"Correspondence", r"Description and travel|Voyages and travels|Travel"):
        d = df[df["Subjects"].str.contains(pat, case=False) & ~df["Text#"].isin(picked)]
        d = d.iloc[rng.permutation(len(d))]
        picked += list(d["Text#"])
    d = df[df["Text#"].isin(picked)].set_index("Text#").loc[picked].reset_index()
    d = d.groupby("Authors", sort=False).head(3)
    ids = [int(x) for x in d["Text#"].head(n_books)]
    json.dump(ids, open(out, "w"))
    return ids


def encounter_rows(words: list[str]) -> dict[str, list[dict]]:
    rows = {w: [] for w in words}
    ids = encounter_book_ids()
    for i, bid in enumerate(ids):
        text = gutenberg_text(bid)
        if not text:
            continue
        for s in split_sentences(text):
            c = clean_sentence(s)
            for w in words:
                if keep_sentence(c, w):
                    rows[w].append({"word": w, "sentence": c, "genre": "encounter",
                                    "source_id": f"gutenberg:{bid}"})
        if (i + 1) % 50 == 0:
            print(f"[harvest] encounter {i + 1}/{len(ids)} books", flush=True)
    return rows


# ---------------------------------------------------------------- sampling
def sample(rows: list[dict], n: int, cap: int | None, rng) -> list[dict]:
    seen, pool = set(), []
    for r in rows:
        key = r["sentence"].lower()
        if key not in seen:
            seen.add(key)
            pool.append(r)
    pool = [pool[i] for i in rng.permutation(len(pool))]
    per_src, out = {}, []
    for r in pool:
        src = r["source_id"]
        if cap and not src.startswith("legacy:") and per_src.get(src, 0) >= cap:
            continue
        per_src[src] = per_src.get(src, 0) + 1
        out.append(r)
        if len(out) >= n:
            break
    return out


def harvest(n_per: int | None = None) -> pd.DataFrame:
    """Build data/natural.csv. Reuses the legacy cache, then fills from the API and Gutenberg."""
    ensure_dirs()
    set_seed()
    cfg = config()["harvest"]
    n_per = n_per or cfg["per_word_per_genre"]
    rng = np.random.default_rng(config()["seed"])
    words = all_words()
    checks, out = [], []

    legacy = {}
    for w in words:
        legacy[w], check = legacy_wikipedia(w)
        if check:
            checks.append(check)
    missing = [w for w in words if len({r["sentence"] for r in legacy[w]}) < n_per]
    if cfg["wiki_source"] == "dump":
        fresh = dump_wikipedia(missing)
    else:
        fresh = {w: fresh_wikipedia(w) for w in missing}
    for w in words:
        rows = legacy[w] + fresh.get(w, [])
        picked = sample(rows, n_per, cfg["max_per_source"]["wikipedia"], rng)
        print(f"[harvest] wikipedia {w}: {len(picked)} (legacy {len(legacy[w])}, pool {len(rows)})")
        out += picked

    fic = fiction_rows(words)
    for w in words:
        picked = sample(fic[w], n_per, cfg["max_per_source"]["fiction"], rng)
        print(f"[harvest] fiction {w}: {len(picked)} (pool {len(fic[w])})")
        out += picked

    df = pd.DataFrame(out)
    df["half"] = df["word"].map(half_of)
    df = df.reset_index(drop=True)
    df.insert(0, "sid", [f"s{i:05d}" for i in range(len(df))])
    df[["sid", "word", "sentence", "genre", "source_id", "half"]].to_csv(
        DATA / "natural.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    if checks:
        pd.DataFrame(checks).to_csv(RESULTS / "legacy_cache_check.csv", index=False)
        for c in checks:
            if c["match"] is False:
                print(f"[harvest] MISMATCH {c['word']}: {c['legacy_sentences']} cached sentences "
                      f"vs {c['legacy_embedding_rows']} old embedding rows "
                      f"({c['legacy_with_target']} contain the target); old embeddings not reused")
    counts = df.groupby(["word", "genre"]).size().unstack(fill_value=0)
    counts.to_csv(RESULTS / "harvest_counts.csv")
    short = counts[(counts < n_per).any(axis=1)]
    if len(short):
        print("[harvest] words below target:\n" + short.to_string())
    return df


def load_natural() -> pd.DataFrame:
    return pd.read_csv(DATA / "natural.csv")


if __name__ == "__main__":
    harvest()
