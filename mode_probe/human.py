"""Human labels for the gold sample: a spreadsheet to fill in, and agreement once filled.

python -m mode_probe.human --sheet    # write data/gold_human_to_label.xlsx (refuses to overwrite labels)
python -m mode_probe.human --report   # kappa of the human labels against the version-3 labels
                                      # (Claude full set, Claude sample, Qwen); results/v3/kappa_human_*.csv

Labels follow LABELING_v3.md (reference x occasion, eight labels), so that human labels can
test the case the two readings disagree on. The sheet shows only the target noun and the
sentence: no model or earlier labels. The label column has a drop-down of the eight labels;
anything else is rejected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATA, PKG, RESULTS, config, target_spans
from .label import GOLD
from .label_v3 import GOLD as GOLD_V3, LABELED as LABELED_V3, LABELS_V3 as LABELS, QWEN as QWEN_V3

SHEET = DATA / "gold_human_to_label.xlsx"

FILL = {"P-ep": "D6E4F2", "P-gen": "C9DCEB", "K-ep": "FDEBD9", "K-gen": "FBE3CC",
        "A-idiom": "E4DFF7", "A-simile": "EEE9FA", "A-def": "D7F0E4", "X": "E6E5E0"}

LEGEND = [
    ("P-ep", "Particular thing(s), one occasion.",
     "The cat jumped off the table and knocked the cup over."),
    ("P-gen", "Particular thing(s), in general: a habit or a standing property.",
     "My cat always sleeps by the fire. / The bridge carries six lanes."),
    ("K-ep", "The kind, one occasion (an episode in its history).",
     "Cats reached Australia in the seventeenth century."),
    ("K-gen", "The kind, in general.", "Cats hunt at night. / Horses are herbivores."),
    ("A-idiom", "Fixed expression; the meaning does not involve the thing.", "He let the cat out of the bag."),
    ("A-simile", "X only as the image in a comparison.", "She moved as quietly as a cat. / It fell like a stone."),
    ("A-def", "States what the thing is.", "A spoon is a utensil with a small shallow bowl."),
    ("X", "Exclude: modifier or material, a verb, a name or title, another sense, or undecidable.",
     "stone wall / built of stone / Apple Inc. / the road forks / a binary tree"),
]

TESTS = [
    "Step 1, first: X, A-idiom, A-simile or A-def, if one applies.",
    "Step 2, reference: could you point to the X the sentence is about, at least in principle?  "
    "yes -> P (particular);  no, it holds whichever Xs exist -> K (kind)",
    "Step 3, occasion: is its point located at a time (an episode, even a long one)?  "
    "yes -> ep;  no, it says how things go or are at any time -> gen",
    "A description inside a narrated scene (\"the lamp smoked\") is ep. "
    "\"Spoons were found in the tomb\" is P: some particular spoons.",
    "",
    f"The full criterion is {PKG.name}/LABELING_v3.md. Label from the sentence alone.",
]


def _rich(sentence: str, word: str):
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont

    spans = target_spans(sentence, word)
    if not spans:
        return sentence
    bold = InlineFont(b=True, color="1F4E79")
    parts, pos = [], 0
    for a, b in spans:
        if a > pos:
            parts.append(sentence[pos:a])
        parts.append(TextBlock(bold, sentence[a:b]))
        pos = b
    if pos < len(sentence):
        parts.append(sentence[pos:])
    return CellRichText(*parts)


def make_sheet(path: Path = SHEET, force: bool = False) -> Path:
    from openpyxl import Workbook
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    if path.exists() and not force and len(read_labels(path)):
        raise SystemExit(f"[human] {path} already has labels; not overwriting")
    gold = pd.read_csv(GOLD, keep_default_na=False)
    rng = np.random.default_rng(config()["seed"] + 11)
    gold = gold.iloc[rng.permutation(len(gold))].reset_index(drop=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Sentences"
    head = ["n", "word", "sentence", "label", "note", "sid"]
    ws.append(head)
    for i, r in enumerate(gold.itertuples(), 1):
        ws.append([i, r.word, None, None, None, r.sid])
        ws.cell(row=i + 1, column=3).value = _rich(r.sentence, r.word)
    last = len(gold) + 1
    for col, w in zip("ABCDEF", (5, 10, 95, 11, 30, 9)):
        ws.column_dimensions[col].width = w
    top = Alignment(vertical="top", wrap_text=True)
    for row in ws.iter_rows(min_row=2, max_row=last):
        for c in row:
            c.alignment = top
        row[5].font = Font(color="9A9A9A", size=8)
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:F{last}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "1:1"

    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    dv.error, dv.errorTitle = "Choose one of: " + ", ".join(LABELS), "Not a label"
    dv.prompt, dv.promptTitle = "P-ep, P-gen, K-ep, K-gen, A-idiom, A-simile, A-def or X", "Label"
    dv.showErrorMessage = dv.showInputMessage = True
    ws.add_data_validation(dv)
    dv.add(f"D2:D{last}")
    for lab, color in FILL.items():
        ws.conditional_formatting.add(
            f"D2:D{last}", CellIsRule(operator="equal", formula=[f'"{lab}"'],
                                      fill=PatternFill("solid", start_color=color, end_color=color)))

    lg = wb.create_sheet("How to label")
    lg.append(["label", "use it when", "examples"])
    for lab, when, ex in LEGEND:
        lg.append([lab, when, ex])
        lg.cell(row=lg.max_row, column=1).fill = PatternFill("solid", start_color=FILL[lab],
                                                             end_color=FILL[lab])
    lg.append([])
    for line in TESTS:
        lg.append([None, line])
    lg.append([])
    lg.append([None, "Labeled so far:", f'=COUNTA(Sentences!D2:D{last})&" of {len(gold)}"'])
    for col, w in zip("ABC", (10, 70, 70)):
        lg.column_dimensions[col].width = w
    for row in lg.iter_rows():
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
    for c in lg[1]:
        c.font = Font(bold=True)
    wb.save(path)
    return path


def read_labels(path: Path = SHEET) -> pd.DataFrame:
    """Filled rows only, as sid,label. Unknown labels raise, so a typo cannot pass silently."""
    if not path.exists():
        return pd.DataFrame(columns=["sid", "label"])
    df = pd.read_excel(path, sheet_name="Sentences", dtype=str, keep_default_na=False)
    df["label"] = df["label"].str.strip()
    df = df[df["label"] != ""]
    bad = df[~df["label"].isin(LABELS)]
    if len(bad):
        raise ValueError(f"unknown labels in rows {bad['n'].tolist()}: {sorted(set(bad['label']))}")
    return df[["sid", "label"]].reset_index(drop=True)


def report(path: Path = SHEET) -> dict:
    human = read_labels(path)
    if human.empty:
        print("[human] no labels yet")
        return {}
    out = {"n": len(human)}
    from sklearn.metrics import cohen_kappa_score

    (RESULTS / "v3").mkdir(parents=True, exist_ok=True)
    others = [("claude_full", LABELED_V3), ("claude_sample", GOLD_V3), ("qwen", QWEN_V3)]
    for name, p in others:
        if p.exists():
            m = human.merge(pd.read_csv(p, keep_default_na=False)[["sid", "label"]], on="sid",
                            suffixes=("_human", "_other"))
            out[name] = float(cohen_kappa_score(m["label_human"], m["label_other"]))
            pd.crosstab(m["label_human"], m["label_other"]).to_csv(RESULTS / "v3" / f"kappa_human_{name}.csv")
    print(f"[human] {len(human)} labels; κ vs " +
          ", ".join(f"{k} {v:.3f}" for k, v in out.items() if k != "n"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.sheet:
        print(f"[human] wrote {make_sheet(force=a.force)}")
    if a.report:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
