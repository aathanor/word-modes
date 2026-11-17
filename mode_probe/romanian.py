"""E8: Romanian sentences for the same twenty nouns, for a person to label.

Romanian attaches the definite article to the end of the noun and generics usually take it
("Pisica este un mamifer"), so the same definite form appears in both modes. A direction learned
on English that still sorts Romanian sentences is not riding on English determiners.

Sources: Romanian Wikipedia (2023-11-01 dump) and Romanian Wikisource (2023-12-01 dump, mostly
literature), from the Wikimedia datasets on Hugging Face. Forms are listed by hand (case, number,
article), in both the current spelling and the older one (î for â, cedilla for comma below).
Ambiguous words: "sticlă" is also glass, "pod" also an attic, "măr" also an apple tree; such uses
are labeled X. "Roșie" (tomato) is also the adjective "red", so "tomată" is used instead.

python -m mode_probe.romanian --sheet     -> data/romanian_to_label.xlsx (15 per word, 300)
python -m mode_probe.romanian --import    -> data/romanian_labeled.csv (rows a person labeled)
then: python -m mode_probe.run_all --labels-ready --romanian
"""
from __future__ import annotations

import argparse
import csv
import re

import numpy as np
import pandas as pd

from .common import DATA, LABELS, RAW, config
from .harvest import _download, clean_sentence, split_sentences

RO = RAW / "romanian"
SHEET = DATA / "romanian_to_label.xlsx"
OUT = DATA / "romanian_labeled.csv"
PER_WORD = 15
FILES = {
    "wikipedia": ["https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ro/train-0000{i}-of-00002.parquet".format(i=i) for i in range(2)],
    "wikisource": ["https://huggingface.co/datasets/wikimedia/wikisource/resolve/main/20231201.ro/train-00000-of-00001.parquet"],
}

FORMS = {
    "cat": ("pisică", "pisică pisica pisicii pisici pisicile pisicilor"),
    "hammer": ("ciocan", "ciocan ciocanul ciocanului ciocane ciocanele ciocanelor"),
    "apple": ("măr", "măr mărul mărului mere merele merelor"),
    "river": ("râu", "râu râul râului râuri râurile râurilor"),
    "chair": ("scaun", "scaun scaunul scaunului scaune scaunele scaunelor"),
    "horse": ("cal", "cal calul calului cai caii cailor"),
    "bottle": ("sticlă", "sticlă sticla sticlei sticle sticlele sticlelor"),
    "bridge": ("pod", "pod podul podului poduri podurile podurilor"),
    "knife": ("cuțit", "cuțit cuțitul cuțitului cuțite cuțitele cuțitelor"),
    "tree": ("copac", "copac copacul copacului copaci copacii copacilor"),
    "dog": ("câine", "câine câinele câinelui câini câinii câinilor"),
    "fork": ("furculiță", "furculiță furculița furculiței furculițe furculițele furculițelor"),
    "tomato": ("tomată", "tomată tomata tomatei tomate tomatele tomatelor"),
    "mountain": ("munte", "munte muntele muntelui munți munții munților"),
    "door": ("ușă", "ușă ușa ușii uși ușile ușilor"),
    "cow": ("vacă", "vacă vaca vacii vaci vacile vacilor"),
    "lamp": ("lampă", "lampă lampa lămpii lămpi lămpile lămpilor"),
    "road": ("drum", "drum drumul drumului drumuri drumurile drumurilor"),
    "spoon": ("lingură", "lingură lingura lingurii linguri lingurile lingurilor"),
    "stone": ("piatră", "piatră piatra pietrei pietre pietrele pietrelor"),
}


def _norm(s: str) -> str:
    """Current spelling: comma-below ș ț, and â inside words where old texts write î."""
    return s.replace("ş", "ș").replace("ţ", "ț").replace("Ş", "Ș").replace("Ţ", "Ț")


def _variants(form: str) -> set[str]:
    out = {form}
    if "â" in form:
        out.add(form.replace("â", "î"))          # older spelling: rîu, cîine
    return out


def form_regex(word_en: str) -> re.Pattern:
    forms = set()
    for f in FORMS[word_en][1].split():
        forms |= _variants(f)
    alt = "|".join(sorted(map(re.escape, forms), key=len, reverse=True))
    return re.compile(rf"(?<![\w-])({alt})(?![\w-])", re.IGNORECASE)


def keep(sentence: str, word_en: str) -> bool:
    toks = sentence.split()
    if not 6 <= len(toks) <= 40 or not sentence.rstrip().endswith((".", "!", "?")):
        return False
    hits = list(form_regex(word_en).finditer(sentence))
    if len(hits) != 1:
        return False
    a = hits[0].start()
    return not (a > 0 and sentence[a].isupper())       # capitalised mid-sentence: part of a name


def pool() -> pd.DataFrame:
    import pyarrow.parquet as pq

    out = RO / "pool.parquet"
    if out.exists():
        return pd.read_parquet(out)
    any_re = re.compile("|".join(form_regex(w).pattern for w in FORMS), re.IGNORECASE)
    rows = []
    for genre, urls in FILES.items():
        for url in urls:
            path = RO / genre / url.rsplit("/", 1)[1]
            _download(url, path)
            for batch in pq.ParquetFile(path).iter_batches(columns=["title", "text"], batch_size=2000):
                for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                    if not text or not any_re.search(text):
                        continue
                    for line in _norm(text).split("\n"):
                        if not any_re.search(line):
                            continue
                        for s in split_sentences(line):
                            c = clean_sentence(s)
                            for w in FORMS:
                                if keep(c, w):
                                    rows.append({"word_en": w, "word": FORMS[w][0], "sentence": c,
                                                 "genre": genre, "source_id": f"{genre}:{title}"})
            print(f"[romanian] {path.name}: {len(rows)} candidates", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("sentence")
    df.to_parquet(out)
    return df


def sample(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(config()["seed"] + 61)
    picks = []
    for w in FORMS:
        sub = df[df.word_en == w]
        per_src: dict = {}
        for genre, n in (("wikisource", (PER_WORD + 1) // 2), ("wikipedia", PER_WORD // 2)):
            g = sub[sub.genre == genre]
            got = 0
            for r in g.iloc[rng.permutation(len(g))].to_dict("records"):
                if per_src.get(r["source_id"], 0) >= 2:
                    continue
                per_src[r["source_id"]] = per_src.get(r["source_id"], 0) + 1
                picks.append(r)
                got += 1
                if got >= n:
                    break
        short = PER_WORD - sum(1 for p in picks if p["word_en"] == w)
        if short > 0:   # top up from either source if one is thin
            rest = sub[~sub.sentence.isin({p["sentence"] for p in picks})]
            picks += rest.iloc[rng.permutation(len(rest))].head(short).to_dict("records")
    out = pd.DataFrame(picks)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    out.insert(0, "sid", [f"r{i:04d}" for i in range(len(out))])
    return out


def make_sheet() -> None:
    from openpyxl import Workbook
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    if SHEET.exists() and len(read_labels()):
        raise SystemExit(f"[romanian] {SHEET} already has labels; not overwriting")
    df = sample(pool())
    wb = Workbook()
    ws = wb.active
    ws.title = "Sentences"
    ws.append(["n", "word", "English", "sentence", "label", "note", "sid", "source"])
    bold = InlineFont(b=True, color="1F4E79")
    for i, r in enumerate(df.itertuples(), 1):
        ws.append([i, r.word, r.word_en, None, None, None, r.sid, r.source_id])
        m = form_regex(r.word_en).search(r.sentence)
        ws.cell(row=i + 1, column=4).value = CellRichText(r.sentence[:m.start()], TextBlock(bold, m.group(0)),
                                                         r.sentence[m.end():])
    last = len(df) + 1
    for col, w in zip("ABCDEFGH", (5, 11, 9, 90, 11, 28, 8, 30)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, max_row=last):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
        row[6].font = row[7].font = Font(color="9A9A9A", size=8)
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "A2"
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    dv.error, dv.errorTitle = "Choose one of: " + ", ".join(LABELS), "Not a label"
    dv.showErrorMessage = True
    ws.add_data_validation(dv)
    dv.add(f"E2:E{last}")
    fill = {"O": "D6E4F2", "A-gen": "FBE3CC", "A-idiom": "E4DFF7", "A-def": "D7F0E4", "X": "E6E5E0"}
    for lab, color in fill.items():
        ws.conditional_formatting.add(f"E2:E{last}", CellIsRule(operator="equal", formula=[f'"{lab}"'],
                                      fill=PatternFill("solid", start_color=color, end_color=color)))
    lg = wb.create_sheet("How to label")
    for row in [["label", "use it when", "examples"],
                ["O", "One particular thing (or several) on one occasion.", "Pisica a sărit de pe masă și a răsturnat cana."],
                ["A-gen", "A literal general statement about the kind, including habits and standing properties.", "Pisicile vânează noaptea."],
                ["A-idiom", "A fixed expression, figure or simile; the meaning does not involve the thing.", "S-a dus ca o piatră în fântână."],
                ["A-def", "States what the thing is.", "Pisica este un mamifer carnivor din familia felidelor."],
                ["X", "Exclude: another sense (sticlă = glass, pod = attic), a modifier or material, a name, or undecidable.", "geam de sticlă; podul casei; Podul Mogoșoaiei"],
                [],
                [None, "The criterion is the one used for English (mode_probe/LABELING.md, version 2). Label from the sentence alone."],
                [None, "Labeled so far:", f'=COUNTA(Sentences!E2:E{last})&" of {len(df)}"']]:
        lg.append(row)
    for col, w in zip("ABC", (9, 70, 60)):
        lg.column_dimensions[col].width = w
    wb.save(SHEET)
    print(f"[romanian] wrote {SHEET}: {len(df)} sentences, "
          f"{df.groupby('genre').size().to_dict()}")


def read_labels() -> pd.DataFrame:
    if not SHEET.exists():
        return pd.DataFrame()
    df = pd.read_excel(SHEET, sheet_name="Sentences", dtype=str, keep_default_na=False)
    df["label"] = df["label"].str.strip()
    df = df[df["label"] != ""]
    bad = df[~df["label"].isin(LABELS)]
    if len(bad):
        raise ValueError(f"unknown labels in rows {bad['n'].tolist()}: {sorted(set(bad['label']))}")
    return df


def import_labels() -> pd.DataFrame:
    df = read_labels()
    out = df.rename(columns={"English": "word_en"})[["sid", "word", "word_en", "sentence", "label", "source"]]
    out.to_csv(OUT, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"[romanian] {len(out)} labeled rows -> {OUT}: {out['label'].value_counts().to_dict()}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--import", dest="imp", action="store_true")
    a = ap.parse_args(argv)
    RO.mkdir(parents=True, exist_ok=True)
    if a.sheet:
        make_sheet()
    if a.imp:
        import_labels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
