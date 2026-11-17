"""Human labels for the gold sample: a spreadsheet to fill in, and agreement once filled.

python -m mode_probe.human --sheet    # write data/gold_human_to_label.xlsx (refuses to overwrite labels)
python -m mode_probe.human --report   # kappa of the human labels against Claude, Qwen and the
                                      # first gold labeling; results/kappa_human_*.csv

The sheet shows only the target noun and the sentence: no model or earlier labels.
The label column has a drop-down of the five labels; anything else is rejected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATA, LABELS, PKG, config, target_spans
from .label import GOLD, LABELED, QWEN_V2, kappa_report

SHEET = DATA / "gold_human_to_label.xlsx"

FILL = {"O": "D6E4F2", "A-gen": "FBE3CC", "A-idiom": "E4DFF7", "A-def": "D7F0E4", "X": "E6E5E0"}

LEGEND = [
    ("O", "One specific instance (or several particular ones) on an occasion.",
     "The cat was asleep on the chair. / The cats scratched me as I put them out."),
    ("A-gen", "Literal general statement about the kind, including habits and procedures.",
     "Cats hunt at night. / Every spring the men climb the trees and shake the branches."),
    ("A-idiom", "Fixed expression, figure or simile; the meaning does not involve the thing.",
     "Let the cat out of the bag. / He fell like a stone."),
    ("A-def", "States what the thing is: \"a/the X is (a) Y\".",
     "A spoon is a utensil with a shallow bowl."),
    ("X", "Exclude: modifier or material, a verb, a name or title, another sense, or undecidable.",
     "stone wall / built of stone / Apple Inc. / the road forks / the chair of the committee"),
]

TESTS = [
    "Apply in order:",
    "1. Could the sentence be true of exactly one X on one day, and is that its point?  ->  O",
    "2. Is X inside a fixed expression or figure (similes included)?  ->  A-idiom",
    "3. Is it of the form \"a/the X is (a) Y\", stating what X is?  ->  A-def",
    "4. Otherwise  ->  A-gen",
    "Before the tests: if X only modifies another noun or names a material, or is another sense "
    "or part of speech, label X.",
    "",
    f"The full criterion, with every clarification, is {PKG.name}/LABELING.md. "
    "Label from the sentence alone.",
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
    dv.prompt, dv.promptTitle = "O, A-gen, A-idiom, A-def or X", "Label"
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
    for col, w in zip("ABC", (9, 70, 70)):
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
    others = [("claude", LABELED), ("qwen", QWEN_V2), ("first_gold", GOLD)]
    for name, p in others:
        if p.exists():
            k = kappa_report(human, pd.read_csv(p, keep_default_na=False), name=f"kappa_human_{name}")
            out[name] = k["overall"]
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
