"""Recover the Wikipedia article for sentences reused from the legacy cache (CC BY-SA
attribution). Pass 1: scan all shards of the 2023-11-01 dump for each sentence.
Pass 2: exact-phrase search on the live API for what is left. Cached in
data/legacy_sources.csv (sid, title, method)."""
from __future__ import annotations

import re
import time

import pandas as pd
import requests

from .common import DATA, RAW, word_regex
from .harvest import DUMP_URL, UA, WIKI_API, _download

OUT = DATA / "legacy_sources.csv"
N_SHARDS = 41
KEEP_SHARDS = 3            # shards the harvest itself uses stay on disk


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _load() -> dict[str, dict]:
    if OUT.exists():
        df = pd.read_csv(OUT, keep_default_na=False)
        return {r["sid"]: r for r in df.to_dict("records")}
    return {}


def _save(found: dict[str, dict]) -> None:
    pd.DataFrame(found.values(), columns=["sid", "title", "method"]).to_csv(OUT, index=False)


def scan_dump(legacy: pd.DataFrame, found: dict) -> None:
    import pyarrow.parquet as pq

    keys = {}            # word -> {key: sid}
    for r in legacy.itertuples():
        if r.sid in found and found[r.sid]["title"]:
            continue
        k = _norm(r.sentence)[:70]
        if len(k) >= 30:
            keys.setdefault(r.word, {})[k] = r.sid
    if not keys:
        return
    regs = {w: word_regex(w) for w in keys}
    for i in range(N_SHARDS):
        if not any(keys.values()):
            break
        path = RAW / "wikipedia" / "dump" / f"train-{i:05d}.parquet"
        _download(DUMP_URL.format(i=i), path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["title", "text"], batch_size=2000):
            for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                for w, kw in keys.items():
                    if not kw or not regs[w].search(text):
                        continue
                    for line in text.split("\n"):
                        if not regs[w].search(line):
                            continue
                        nl = _norm(line)
                        for k in [k for k in kw if k in nl]:
                            sid = kw.pop(k)
                            found[sid] = {"sid": sid, "title": title, "method": "dump-2023-11-01"}
        _save(found)
        left = sum(len(v) for v in keys.values())
        print(f"[attribution] shard {i + 1}/{N_SHARDS}: {len(found)} found, {left} left", flush=True)
        if i >= KEEP_SHARDS:
            path.unlink(missing_ok=True)


def _search(sentence: str) -> str:
    words = [w.strip(',.;:()"“”‘’') for w in sentence.split()]
    phrase = " ".join(w for w in words[:9] if w)
    for attempt in range(6):
        try:
            r = requests.get(WIKI_API, headers=UA, timeout=30, params={
                "action": "query", "list": "search", "srsearch": f'"{phrase}"',
                "srlimit": 1, "srnamespace": 0, "format": "json", "formatversion": 2})
            if r.status_code == 429 or "json" not in r.headers.get("content-type", ""):
                time.sleep(int(r.headers.get("retry-after") or 0) or 20 * (attempt + 1))
                continue
            hits = r.json()["query"]["search"]
            return hits[0]["title"] if hits else ""
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(10 * (attempt + 1))
    return ""


def search_rest(legacy: pd.DataFrame, found: dict) -> None:
    todo = [r for r in legacy.itertuples() if r.sid not in found]
    for i, r in enumerate(todo):
        title = _search(r.sentence)
        found[r.sid] = {"sid": r.sid, "title": title, "method": "api-search" if title else "not found"}
        time.sleep(3.0)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            _save(found)
            print(f"[attribution] api {i + 1}/{len(todo)}", flush=True)


def recover() -> pd.DataFrame:
    nat = pd.read_csv(DATA / "natural.csv")
    legacy = nat[nat["source_id"].str.startswith("legacy:")]
    found = _load()
    scan_dump(legacy, found)
    search_rest(legacy, found)
    _save(found)
    df = pd.DataFrame(found.values())
    print(df["method"].value_counts().to_string())
    return df


if __name__ == "__main__":
    recover()
