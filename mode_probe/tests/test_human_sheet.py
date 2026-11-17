import pandas as pd
import pytest
from openpyxl import load_workbook

from mode_probe import human


def test_sheet_roundtrip(tmp_path):
    p = human.make_sheet(tmp_path / "h.xlsx")
    wb = load_workbook(p, rich_text=True)
    ws = wb["Sentences"]
    assert [c.value for c in ws[1]] == ["n", "word", "sentence", "label", "note", "sid"]
    assert ws.max_row == 301
    dv = ws.data_validations.dataValidation
    assert len(dv) == 1 and "A-idiom" in dv[0].formula1 and "D2:D301" in str(dv[0].sqref)
    assert human.read_labels(p).empty

    ws["D2"], ws["D3"], ws["D4"] = "P-ep", " K-gen ", ""
    wb.save(p)
    got = human.read_labels(p)
    assert got["label"].tolist() == ["P-ep", "K-gen"]
    assert got["sid"].tolist() == [ws["F2"].value, ws["F3"].value]
    with pytest.raises(SystemExit):
        human.make_sheet(p)

    ws["D5"] = "Kgen"
    wb.save(p)
    with pytest.raises(ValueError):
        human.read_labels(p)


def test_sheet_hides_earlier_labels(tmp_path):
    p = human.make_sheet(tmp_path / "h.xlsx")
    df = pd.read_excel(p, sheet_name="Sentences", dtype=str, keep_default_na=False)
    assert (df["label"] == "").all()
    gold = pd.read_csv(human.GOLD, keep_default_na=False)
    assert set(df["sid"]) == set(gold["sid"])
