from __future__ import annotations
import csv
from pathlib import Path

import pandas as pd


""" This helper file adds the relevant English loanwords to the 
CSV file for data enrichment
from lines 301 to 376, altogether 76 lines English
lemmas are added """


def _read_selected_lemmas_from_line_range(
    grouped_csv: str | Path,
    *,
    start_line: int = 301,
    end_line: int = 376,
) -> list[tuple[str, str]]:
    """Read (lemma, pos) pairs from grouped CSV using 1-based line numbers."""
    path = Path(grouped_csv)
    header: list[str] = []
    results: list[tuple[str, str]] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        row = next(csv.reader([line]))
        if not header:
            header = [col.strip().lower() for col in row]
            continue
        if line_no < start_line or line_no > end_line:
            continue
        if not row or not row[0].strip():
            continue
        lemma = row[0].strip()
        pos = ""
        if "pos" in header:
            idx = header.index("pos")
            if idx < len(row):
                pos = row[idx].strip()
        results.append((lemma, pos))
    return results


def append_wiktionary_english_loans(
    *,
    input_csv: str | Path,
    output_csv: str | Path,
    grouped_csv: str | Path,
    start_line: int = 301,
    end_line: int = 376,
    expected_count: int | None = 76,
    skip_existing: bool = True,
) -> pd.DataFrame:

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv, encoding="utf-8-sig", dtype=str).fillna("")

    selected = _read_selected_lemmas_from_line_range(
        grouped_csv, start_line=start_line, end_line=end_line,
    )
    
    seen = set()
    deduped = []
    for item in selected:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    selected = deduped

    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(
            f"Expected {expected_count} selected Wiktionary lemmas, "
            f"but found {len(selected)} in lines {start_line}--{end_line}."
        )

    existing = set(df["lemma"].astype(str).str.strip().str.casefold()) if "lemma" in df.columns else set()
    rows = []
    skipped_existing = []

    for lemma, pos_value in selected:
        key = lemma.strip().casefold()
        if skip_existing and key in existing:
            skipped_existing.append(lemma)
            continue

        row = {col: "" for col in df.columns}
        row.update({
            "lemma": lemma,
            "homograph": "",
            "ipa": "",
            "pos": pos_value,
            "pos_source": "wiktionary",
            "is_dialectal": "no",
            "is_archaic": "no",
            "is_obsolete": "no",
            "origin_type": "loan",
            "formation_type": "simplex",
            "primary_donor_language": "English",
            "primary_donor_family": "English",
            "entry_type": "single",
            "entry_file": "wiktionary",
        })
        rows.append(row)

    out = pd.concat([df, pd.DataFrame(rows, columns=df.columns)], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Read selected Wiktionary lemmas: {len(selected)}")
    print(f"Appended rows: {len(rows)}")
    if skipped_existing:
        print(f"Skipped existing lemmas: {len(skipped_existing)}")
        print(", ".join(skipped_existing[:30]))
    print(f"Saved: {output_csv}")
    return out


__all__ = ["append_wiktionary_english_loans"]