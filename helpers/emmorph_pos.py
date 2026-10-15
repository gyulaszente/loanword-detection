"""Add emMorph POS tags to the compact UESz CSV export.
THIS file also includes the fallback steps for POS-tagging
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import platform


_hfst_bin = "hfst-lookup.exe" if platform.system() == "Windows" else "hfst-lookup"
_DEFAULT_HFST_LOOKUP = shutil.which(_hfst_bin) or shutil.which("hfst-lookup")
DEFAULT_HFST_LOOKUP = Path(os.environ.get("HFST_LOOKUP", _DEFAULT_HFST_LOOKUP or _hfst_bin))
DEFAULT_HU_HFSTOL = Path(
    os.environ.get(
        "HU_HFSTOL",
        Path(__file__).parent.parent / "input_files" / "hu.hfstol",
    )
)

INTEGERISH_COLUMNS = ["first_year", "certainty_level"]

SEGMENT_RE = re.compile(r"([^[]+)\[([^\]]+)\]")
POS_EXCLUDED_CASE_TAGS = {"Acc", "Ess"}
UPOS_ORDER = ["NOUN", "VERB", "ADJ", "ADV", "INTJ", "DET", "NUM", "ADP", "CCONJ", "PART",]


def add_pos_to_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    hfst_lookup: str | Path = DEFAULT_HFST_LOOKUP,
    hu_hfstol: str | Path = DEFAULT_HU_HFSTOL,
    batch_size: int = 1000,
    lemma_column: str = "lemma",
    homograph_column: str = "homograph",
    definition_column: str = "definition_hun",
    translation_column: str = "translation",
    german_pos_tagger: Callable[[list[str]], dict[str, str | None]] | None = None,
    keep_debug_columns: bool = False,
    strict_surface_match: bool = True,
) -> pd.DataFrame:

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    hfst_lookup = Path(hfst_lookup)
    hu_hfstol = Path(hu_hfstol)
    _check_emmorph_paths(hfst_lookup, hu_hfstol)

    df = pd.read_csv(input_csv, encoding="utf-8-sig", dtype=str).fillna("")
    if lemma_column not in df.columns:
        raise ValueError(f"Missing lemma column {lemma_column!r} in {input_csv}")

    lemmas = sorted({
        normalize_lemma(value)
        for value in df[lemma_column]
        if normalize_lemma(value)
    })
    print(f"Rows: {len(df)}")
    print(f"Unique lemmas: {len(lemmas)}")

    lemma_to_pos: dict[str, str | None] = {}
    for start in range(0, len(lemmas), batch_size):
        batch = lemmas[start:start + batch_size]
        print(f"Analysing batch {start}-{start + len(batch) - 1}")
        analyses = emmorph_analyse_many(
            batch,
            hfst_lookup=hfst_lookup,
            hu_hfstol=hu_hfstol,
            strict_surface_match=strict_surface_match,
        )
        for lemma in batch:
            lemma_to_pos[lemma] = all_upos_for_lemma(lemma, analyses.get(lemma, []))

    definition_to_pos: dict[str, str | None] = {}
    if definition_column in df.columns:
        definition_glosses = sorted({
            gloss
            for value in df[definition_column]
            for gloss in _one_word_pipe_values(value)
        })
        for start in range(0, len(definition_glosses), batch_size):
            batch = definition_glosses[start:start + batch_size]
            print(f"Analysing definition batch {start}-{start + len(batch) - 1}")
            analyses = emmorph_analyse_many(
                batch,
                hfst_lookup=hfst_lookup,
                hu_hfstol=hu_hfstol,
                strict_surface_match=strict_surface_match,
            )
            for gloss in batch:
                definition_to_pos[gloss] = all_upos_for_lemma(gloss, analyses.get(gloss, []))

    translation_to_pos: dict[str, str | None] = {}
    if german_pos_tagger is not None and translation_column in df.columns:
        translation_glosses = sorted({
            gloss
            for value in df[translation_column]
            for gloss in _one_word_pipe_values(value)
        })
        if translation_glosses:
            tagged = german_pos_tagger(translation_glosses)
            translation_to_pos = {
                gloss: _normalize_upos(str(tagged.get(gloss) or ""))
                for gloss in translation_glosses
            }


    """ FALLBACK ARRAYS FOR THE POS TAGS """
    out = df.copy()
    final_pos: list[str] = []
    pos_source: list[str] = []
    pos_emmorph_lemma: list[str] = []
    pos_fallback_hun: list[str] = []
    pos_fallback_hun_gloss: list[str] = []
    pos_fallback_ger: list[str] = []
    pos_fallback_ger_gloss: list[str] = []

    for _, row in out.iterrows():
        lemma = normalize_lemma(row.get(lemma_column))
        lemma_pos = lemma_to_pos.get(lemma) or ""
        is_homograph_entry = _has_homograph_value(row.get(homograph_column))

        chosen_pos = lemma_pos
        chosen_source = "emmorph_lemma" if lemma_pos else ""
        hun_pos = ""
        hun_gloss = ""
        ger_pos = ""
        ger_gloss = ""
        hun_glosses = _one_word_pipe_values(row.get(definition_column)) if definition_column in out.columns else []
        ger_glosses = _one_word_pipe_values(row.get(translation_column)) if translation_column in out.columns else []
        
        if lemma_pos and is_homograph_entry and _is_ambiguous_pos(chosen_pos):
            candidates = set(_split_pos_labels(chosen_pos))
            fallback_hun_pos = None
            supporting_hun_glosses: list[str] = []

            if len(hun_glosses) > 1:
                fallback_hun_pos, supporting_hun_glosses = _collect_fallback_pos_labels(
                    hun_glosses,
                    definition_to_pos,
                    candidates=candidates,
                )
                if fallback_hun_pos:
                    chosen_pos = fallback_hun_pos
                    chosen_source = "emmorph_lemma_hun_gloss"
                    hun_pos = fallback_hun_pos
                    hun_gloss = "|".join(supporting_hun_glosses)

            if len(hun_glosses) <= 1 or not fallback_hun_pos:
                fallback_hun_any, supporting_hun_any = _collect_fallback_pos_labels(
                    hun_glosses,
                    definition_to_pos,
                    candidates=candidates,
                )
                fallback_ger_pos, supporting_ger_glosses = _collect_fallback_pos_labels(
                    ger_glosses,
                    translation_to_pos,
                    candidates=candidates,
                )
                combined_pos = _merge_pos_label_strings(fallback_hun_any, fallback_ger_pos)
                if combined_pos:
                    chosen_pos = combined_pos
                    chosen_source = _combined_gloss_source(
                        "emmorph_lemma",
                        supporting_hun_any,
                        supporting_ger_glosses,
                    )
                    if fallback_hun_any:
                        hun_pos = fallback_hun_any
                        hun_gloss = "|".join(supporting_hun_any)
                    if fallback_ger_pos:
                        ger_pos = fallback_ger_pos
                        ger_gloss = "|".join(supporting_ger_glosses)

        if not chosen_pos:
            fallback_hun_pos = None
            supporting_hun_glosses = []
            support_labels = _fallback_unambiguous_support_labels(
                (hun_glosses, definition_to_pos),
                (ger_glosses, translation_to_pos),
            )

            if len(hun_glosses) > 1:
                fallback_hun_pos, supporting_hun_glosses = _collect_fallback_pos_labels(
                    hun_glosses,
                    definition_to_pos,
                    support_labels=support_labels,
                )
                if fallback_hun_pos:
                    chosen_pos = fallback_hun_pos
                    chosen_source = "emmorph_hun_gloss"
                    hun_pos = fallback_hun_pos
                    hun_gloss = "|".join(supporting_hun_glosses)

            if len(hun_glosses) <= 1 or not fallback_hun_pos:
                fallback_hun_any, supporting_hun_any = _collect_fallback_pos_labels(
                    hun_glosses,
                    definition_to_pos,
                    support_labels=support_labels,
                )
                fallback_ger_pos, supporting_ger_glosses = _collect_fallback_pos_labels(
                    ger_glosses,
                    translation_to_pos,
                    support_labels=support_labels,
                )
                combined_pos = _merge_pos_label_strings(fallback_hun_any, fallback_ger_pos)
                if combined_pos:
                    chosen_pos = combined_pos
                    if fallback_hun_any and fallback_ger_pos:
                        chosen_source = "emmorph_hun_gloss_combined_gloss"
                    elif fallback_hun_any:
                        chosen_source = "emmorph_hun_gloss"
                    else:
                        chosen_source = "german_gloss_fallback"
                    if fallback_hun_any:
                        hun_pos = fallback_hun_any
                        hun_gloss = "|".join(supporting_hun_any)
                    if fallback_ger_pos:
                        ger_pos = fallback_ger_pos
                        ger_gloss = "|".join(supporting_ger_glosses)

        final_pos.append(chosen_pos)
        pos_source.append(chosen_source)
        pos_emmorph_lemma.append(lemma_pos)
        pos_fallback_hun.append(hun_pos)
        pos_fallback_hun_gloss.append(hun_gloss)
        pos_fallback_ger.append(ger_pos)
        pos_fallback_ger_gloss.append(ger_gloss)

    out["pos"] = final_pos
    out["pos_source"] = pos_source
    out["pos_emmorph_lemma"] = pos_emmorph_lemma
    out["pos_fallback_hun"] = pos_fallback_hun
    out["pos_fallback_hun_gloss"] = pos_fallback_hun_gloss
    out["pos_fallback_ger"] = pos_fallback_ger
    out["pos_fallback_ger_gloss"] = pos_fallback_ger_gloss
    out = normalize_integerish_columns(out)
    debug_columns = [
        "pos_emmorph_lemma",
        "pos_fallback_hun",
        "pos_fallback_hun_gloss",
        "pos_fallback_ger",
        "pos_fallback_ger_gloss",
    ]
    insert_after = "homograph" if "homograph" in out.columns else lemma_column
    ordered_columns = ["pos", "pos_source"] + (debug_columns if keep_debug_columns else [])
    out = move_columns_after(out, ordered_columns, after=insert_after)
    if not keep_debug_columns:
        out = out.drop(columns=debug_columns, errors="ignore")

    print(out["pos"].value_counts(dropna=False).head(10))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("Saved with POS to:", output_csv)
    return out


def add_emmorph_found_column(
    df: pd.DataFrame,
    *,
    pos_column: str = "pos",
    output_column: str = "emmorph_found",
) -> pd.DataFrame:
    """Return a copy with a boolean-ish column marking rows where lemma emMorph found a POS."""
    out = df.copy()
    source_column = "pos_emmorph_lemma" if "pos_emmorph_lemma" in out.columns else pos_column
    if source_column not in out.columns:
        raise ValueError(f"Missing POS column {source_column!r}")
    out[output_column] = out[source_column].fillna("").astype(str).str.strip() != ""
    return out


def emmorph_coverage_summary(
    df: pd.DataFrame,
    *,
    lemma_column: str = "lemma",
    pos_column: str = "pos",
) -> pd.DataFrame:

    if lemma_column not in df.columns:
        raise ValueError(f"Missing lemma column {lemma_column!r}")
    if pos_column not in df.columns:
        raise ValueError(f"Missing POS column {pos_column!r}")

    out = add_emmorph_found_column(df, pos_column=pos_column)
    rows_total = len(out)
    rows_found = int(out["emmorph_found"].sum())
    rows_missing = rows_total - rows_found

    unique = out[[lemma_column, "emmorph_found"]].copy()
    unique[lemma_column] = unique[lemma_column].astype(str).str.strip().str.lower()
    unique = unique[unique[lemma_column] != ""]
    unique_total = unique[lemma_column].nunique()
    unique_found = unique.loc[unique["emmorph_found"], lemma_column].nunique()
    unique_missing = unique_total - unique_found

    summary = pd.DataFrame([{
        "rows_total": rows_total,
        "rows_with_pos": rows_found,
        "rows_without_pos": rows_missing,
        "rows_with_pos_pct": round((rows_found / rows_total * 100), 2) if rows_total else 0.0,
        "unique_lemmas_total": unique_total,
        "unique_lemmas_with_pos": unique_found,
        "unique_lemmas_without_pos": unique_missing,
        "unique_lemmas_with_pos_pct": round((unique_found / unique_total * 100), 2) if unique_total else 0.0,
    }])
    return summary


def emmorph_found_vs_missing_table(
    df: pd.DataFrame,
    *,
    pos_column: str = "pos",
    flag_columns: tuple[str, ...] = ("is_archaic", "is_obsolete", "is_dialectal"),
    year_column: str = "first_year",
) -> pd.DataFrame:

    out = add_emmorph_found_column(df, pos_column=pos_column)
    for column in flag_columns:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str).str.strip().str.lower().isin({"yes", "true", "1"})

    if year_column not in out.columns:
        out[year_column] = ""
    out["_year_num"] = pd.to_numeric(out[year_column], errors="coerce")

    rows = []
    for found_value, label in [(True, "found"), (False, "missing")]:
        sub = out[out["emmorph_found"] == found_value].copy()
        row = {
            "emmorph_status": label,
            "rows": len(sub),
            "archaic_rows": int(sub["is_archaic"].sum()) if "is_archaic" in sub.columns else 0,
            "obsolete_rows": int(sub["is_obsolete"].sum()) if "is_obsolete" in sub.columns else 0,
            "dialectal_rows": int(sub["is_dialectal"].sum()) if "is_dialectal" in sub.columns else 0,
            "first_year_known": int(sub["_year_num"].notna().sum()),
            "first_year_min": int(sub["_year_num"].min()) if sub["_year_num"].notna().any() else None,
            "first_year_median": int(sub["_year_num"].median()) if sub["_year_num"].notna().any() else None,
            "first_year_max": int(sub["_year_num"].max()) if sub["_year_num"].notna().any() else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def emmorph_pos_value_counts(
    df: pd.DataFrame,
    *,
    pos_column: str = "pos",
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Return a tidy value-count table of POS labels such as NOUN, VERB, NOUN;ADJ."""
    if pos_column not in df.columns:
        raise ValueError(f"Missing POS column {pos_column!r}")
    values = df[pos_column].fillna("").astype(str).str.strip()
    if drop_missing:
        values = values[values != ""]
    counts = values.replace("", "None").value_counts(dropna=False).rename_axis("pos").reset_index(name="count")
    return counts


def emmorph_year_period_table(
    df: pd.DataFrame,
    *,
    pos_column: str = "pos",
    year_column: str = "first_year",
    labels: tuple[str, ...] = (
        "before 1300",
        "1301-1500",
        "1501-1700",
        "1701-1800",
        "1801-1900",
        "1901-1945",
        "1946 and after",
    ),
    bins: tuple[float, ...] = (-float("inf"), 1300, 1500, 1700, 1800, 1900, 1945, float("inf")),
) -> pd.DataFrame:
    """Return a found/missing by first-year period table."""
    out = add_emmorph_found_column(df, pos_column=pos_column)
    out["_year_num"] = pd.to_numeric(out.get(year_column), errors="coerce")
    out["first_year_period"] = pd.cut(out["_year_num"], bins=bins, labels=labels, right=True)
    out["emmorph_status"] = np.where(out["emmorph_found"], "found", "missing")

    table = pd.pivot_table(
        out,
        index="emmorph_status",
        columns="first_year_period",
        values="lemma" if "lemma" in out.columns else pos_column,
        aggfunc="count",
        fill_value=0,
        observed=False,
    )
    table = table.reindex(columns=list(labels), fill_value=0)
    unknown_counts = (
        out[out["_year_num"].isna()]
        .groupby("emmorph_status")
        .size()
        .reindex(["found", "missing"], fill_value=0)
    )
    table["unknown_year"] = unknown_counts
    return table.reset_index()


def emmorph_year_period_percent_table(
    df: pd.DataFrame,
    *,
    pos_column: str = "pos",
    year_column: str = "first_year",
    labels: tuple[str, ...] = (
        "before 1300",
        "1301-1500",
        "1501-1700",
        "1701-1800",
        "1801-1900",
        "1901-1945",
        "1946 and after",
    ),
    bins: tuple[float, ...] = (-float("inf"), 1300, 1500, 1700, 1800, 1900, 1945, float("inf")),
    decimals: int = 2,
) -> pd.DataFrame:

    counts = emmorph_year_period_table(
        df,
        pos_column=pos_column,
        year_column=year_column,
        labels=labels,
        bins=bins,
    ).copy()

    period_columns = list(labels) + ["unknown_year"]
    row_totals = counts[period_columns].sum(axis=1).replace(0, np.nan)
    counts[period_columns] = counts[period_columns].div(row_totals, axis=0).mul(100).round(decimals)
    return counts


def emmorph_flag_found_table(
    df: pd.DataFrame,
    *,
    pos_column: str = "pos",
    flag_columns: tuple[str, ...] = ("is_obsolete", "is_archaic", "is_dialectal"),
) -> pd.DataFrame:

    out = add_emmorph_found_column(df, pos_column=pos_column)
    rows: list[dict[str, Any]] = []

    for column in flag_columns:
        if column not in out.columns:
            out[column] = ""
        mask = out[column].fillna("").astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
        flagged = out[mask]
        found = int(flagged["emmorph_found"].sum())
        missing = int((~flagged["emmorph_found"]).sum())
        total = len(flagged)
        rows.append({
            "flag": column.removeprefix("is_"),
            "found": found,
            "missing": missing,
            "total": total,
            "found_pct": round((found / total * 100), 2) if total else 0.0,
            "missing_pct": round((missing / total * 100), 2) if total else 0.0,
        })

    return pd.DataFrame(rows)


def emmorph_pos_source_flag_table(
    df: pd.DataFrame,
    *,
    pos_source_column: str = "pos_source",
    flag_columns: tuple[str, ...] = ("is_obsolete", "is_archaic", "is_dialectal"),
    percentages: bool = False,
) -> pd.DataFrame:

    out = df.copy()
    if pos_source_column not in out.columns:
        raise ValueError(f"Missing POS source column {pos_source_column!r}")

    flagged = pd.Series(False, index=out.index)
    for column in flag_columns:
        if column not in out.columns:
            continue
        flagged = flagged | out[column].fillna("").astype(str).str.strip().str.lower().isin(
            {"yes", "true", "1"}
        )

    out["_flag_status"] = np.where(flagged, "Flagged", "Not flagged")
    out["_pos_source_group"] = out[pos_source_column].map(_pos_source_group)

    source_order = [
        "emMorph lemma",
        "emMorph Hungarian definition",
        "German translation fallback",
        "Not recognized",
    ]
    flag_order = ["Not flagged", "Flagged"]
    table = pd.crosstab(out["_pos_source_group"], out["_flag_status"])
    table = table.reindex(index=source_order, columns=flag_order, fill_value=0)
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)

    if not percentages:
        return table.reset_index(names="pos_source_group")

    pct = table.loc[source_order, flag_order].div(table.loc["Total", flag_order], axis=1).mul(100)
    pct["Total"] = table.loc[source_order, "Total"].div(table.loc["Total", "Total"]).mul(100)
    pct.loc["Total"] = [100.0, 100.0, 100.0]
    return pct.round(1).reset_index(names="pos_source_group")


def emmorph_pos_cumulative_stage_table(
    df: pd.DataFrame,
    *,
    pos_source_column: str = "pos_source",
    flag_columns: tuple[str, ...] = ("is_obsolete", "is_archaic", "is_dialectal"),
) -> tuple[pd.DataFrame, pd.DataFrame]:

    out = df.copy()
    if pos_source_column not in out.columns:
        raise ValueError(f"Missing POS source column {pos_source_column!r}")

    flagged = pd.Series(False, index=out.index)
    for column in flag_columns:
        if column not in out.columns:
            continue
        flagged = flagged | out[column].fillna("").astype(str).str.strip().str.lower().isin(
            {"yes", "true", "1"}
        )

    source = out[pos_source_column].fillna("").astype(str).str.strip()

    stage_masks = {
        "After emMorph lemma": source.str.startswith("emmorph_lemma"),
        "After Hungarian definition fallback": (
            source.str.startswith("emmorph_lemma")
            | source.str.startswith("emmorph_hun_gloss")
        ),
        "After German translation fallback": source != "",
    }

    row_labels = list(stage_masks) + ["Unrecognized after all steps", "Total"]
    flag_order = ["Not flagged", "Flagged", "Total"]

    total_not_flagged = int((~flagged).sum())
    total_flagged = int(flagged.sum())
    total_all = len(out)

    count_rows: list[dict[str, int]] = []
    for label, mask in stage_masks.items():
        nf = int((mask & ~flagged).sum())
        fl = int((mask & flagged).sum())
        count_rows.append({"Not flagged": nf, "Flagged": fl, "Total": nf + fl})

    unrecognized_mask = source == ""
    nf = int((unrecognized_mask & ~flagged).sum())
    fl = int((unrecognized_mask & flagged).sum())
    count_rows.append({"Not flagged": nf, "Flagged": fl, "Total": nf + fl})
    count_rows.append({"Not flagged": total_not_flagged, "Flagged": total_flagged, "Total": total_all})

    count_table = pd.DataFrame(count_rows, index=row_labels)[flag_order]

    pct_table = count_table.copy().astype(float)
    col_totals = {"Not flagged": total_not_flagged, "Flagged": total_flagged, "Total": total_all}
    for col, denom in col_totals.items():
        pct_table[col] = (count_table[col] / denom * 100).round(1) if denom else 0.0

    return count_table, pct_table


def emmorph_pos_source_detail_table(
    df: pd.DataFrame,
    *,
    pos_source_column: str = "pos_source",
    flag_columns: tuple[str, ...] = ("is_obsolete", "is_archaic", "is_dialectal"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a cumulative detailed table for each concrete 'pos_source' value.

    Returns '(count_table, pct_table)' where percentages are column-normalized
    by flag status and total row count.
    """
    out = df.copy()
    if pos_source_column not in out.columns:
        raise ValueError(f"Missing POS source column {pos_source_column!r}")

    flagged = pd.Series(False, index=out.index)
    for column in flag_columns:
        if column not in out.columns:
            continue
        flagged = flagged | out[column].fillna("").astype(str).str.strip().str.lower().isin(
            {"yes", "true", "1"}
        )

    flag_order = ["Not flagged", "Flagged"]
    source = out[pos_source_column].fillna("").astype(str).str.strip()
    source_order = [
        "emmorph_lemma",
        "emmorph_lemma_hun_gloss",
        "emmorph_lemma_ger_gloss",
        "emmorph_lemma_combined_gloss",
        "emmorph_hun_gloss",
        "emmorph_hun_gloss_combined_gloss",
        "german_gloss_fallback",
    ]
    extra_sources = sorted(
        value for value in source.unique()
        if value and value not in source_order
    )
    source_order.extend(extra_sources)

    total_not_flagged = int((~flagged).sum())
    total_flagged = int(flagged.sum())
    total_all = len(out)

    rows: list[dict[str, int]] = []
    row_labels: list[str] = []
    cumulative_mask = pd.Series(False, index=out.index)
    for label in source_order:
        cumulative_mask = cumulative_mask | (source == label)
        nf = int((cumulative_mask & ~flagged).sum())
        fl = int((cumulative_mask & flagged).sum())
        rows.append({"Not flagged": nf, "Flagged": fl, "Total": nf + fl})
        row_labels.append(f"After {label}")

    unrecognized_mask = source == ""
    nf = int((unrecognized_mask & ~flagged).sum())
    fl = int((unrecognized_mask & flagged).sum())
    rows.append({"Not flagged": nf, "Flagged": fl, "Total": nf + fl})
    row_labels.append("Unrecognized after all steps")
    rows.append({"Not flagged": total_not_flagged, "Flagged": total_flagged, "Total": total_all})
    row_labels.append("Total")

    table = pd.DataFrame(rows, index=row_labels)[["Not flagged", "Flagged", "Total"]]

    pct = table.copy().astype(float)
    col_totals = {"Not flagged": total_not_flagged, "Flagged": total_flagged, "Total": total_all}
    for col, denom in col_totals.items():
        pct[col] = (table[col] / denom * 100).round(1) if denom else 0.0

    return table, pct


def _pos_source_group(value: Any) -> str:
    """Collapse detailed POS source labels into the main assignment stages."""
    text = str(value or "").strip()
    if not text:
        return "Not recognized"
    if text.startswith("emmorph_lemma"):
        return "emMorph lemma"
    if text.startswith("emmorph_hun_gloss"):
        return "emMorph Hungarian definition"
    if text.startswith("german_gloss_fallback"):
        return "German translation fallback"
    return text


def emmorph_analyse_many(
    words: Iterable[str],
    *,
    hfst_lookup: str | Path = DEFAULT_HFST_LOOKUP,
    hu_hfstol: str | Path = DEFAULT_HU_HFSTOL,
    strict_surface_match: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Analyse many surface words in one 'hfst-lookup' call."""
    words = [word for word in words if word]
    if not words:
        return {}

    proc = subprocess.run(
        [
            str(hfst_lookup),
            "--pipe-mode=input",
            "--cascade=composition",
            str(hu_hfstol),
        ],
        input="\n".join(words) + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "emMorph/hfst-lookup failed with exit code "
            f"{proc.returncode}:\n{proc.stderr.strip()}"
        )

    results: dict[str, list[dict[str, Any]]] = {word: [] for word in words}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        surface = parts[0]
        analysis = parts[1]
        weight = float(parts[2].replace(",", ".")) if len(parts) >= 3 and parts[2] else None
        if surface in results:
            results[surface].append({
                "surface": surface,
                "analysis": analysis,
                "weight": weight,
            })
    if strict_surface_match:
        results = _keep_strict_surface_matches(results)
    return results


def _reconstruct_analysis_surface(analysis: str) -> str:
    """
    Return the concatenated segment surfaces from an emMorph analysis.
    Handles emMorph's -ik suffix doubling: verbs like okoskodik are analysed as
    …kodik[_Vbz:kOd/V]ik[Prs.NDef.3Sg], where the suffix morpheme surface
    already ends in -ik and emMorph appends a separate ik[Prs.NDef.3Sg]
    segment, producing a  double: 'okoskodikik'.
    When the last segment is bare 'ik' with a present-tense verbal inflection
    tag, the preceding segment is verbal, and the reconstruction without it
    already ends in 'ik', the trailing segment is dropped. --> 'okoskodik'
    """
    segments = SEGMENT_RE.findall(analysis or "")
    if not segments:
        return ""
    surfaces = [s for s, _ in segments]
    tags    = [t for _, t in segments]

    # -ik doubling check, restricted to verbal present-tense analyses.
    if (surfaces[-1] == "ik"
            and len(segments) > 1
            and tags[-1].startswith("Prs.")
            and "/V" in tags[-2]):
        without_last = normalize_lemma("".join(surfaces[:-1]))
        if without_last.endswith("ik"):
            return without_last

    return normalize_lemma("".join(surfaces))


def _keep_strict_surface_matches(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep only analyses whose segment surfaces reconstruct the input word exactly """
    filtered: dict[str, list[dict[str, Any]]] = {}
    for word, analyses in results.items():
        filtered[word] = [
            item for item in analyses
            if _reconstruct_analysis_surface(str(item.get("analysis", ""))) == normalize_lemma(word)
        ]
    return filtered


def all_upos_for_lemma(lemma: str, analyses: list[dict[str, Any]]) -> str | None:
    """
    Return useful UPOS readings whose extracted root exactly matches 'lemma'.
    Accusative and essive analyses are ignored for POS annotation because the
    dataset contains dictionary lemmas, not inflected token forms
    """
    lemma = normalize_lemma(lemma)
    upos_set = set()

    for item in analyses:
        analysis = str(item.get("analysis", ""))
        if _has_excluded_pos_case_tag(analysis):
            continue
        root, upos = extract_root_and_upos(analysis)
        if normalize_lemma(root) != lemma:
            continue
        if upos is None:
            continue
        upos_set.add(upos)

    if not upos_set:
        return None
    ordered = sorted(upos_set, key=lambda tag: UPOS_ORDER.index(tag) if tag in UPOS_ORDER else len(UPOS_ORDER))
    return ";".join(ordered)


def _has_excluded_pos_case_tag(analysis: str) -> bool:
    """Return true if an emMorph analysis is an excluded inflected-case reading."""
    return any(tag in POS_EXCLUDED_CASE_TAGS for _, tag in SEGMENT_RE.findall(analysis or ""))


def extract_root_and_upos(analysis: str, lemma: str | None = None) -> tuple[str | None, str | None]:
    """
    Extract a root-like surface and coarse UPOS from one emMorph analysis.

    Affix/morpheme segments with tags beginning '_' are skipped.
    """
    if lemma is not None:
        root, upos = _scan_analysis(analysis, require_surface=lemma)
        if upos is not None:
            return root, upos
    return _scan_analysis(analysis)



""" THIS METHOD NORMALIZES THE EMMORPH TAGS INTO REGULAR POS TAGS """
def _scan_analysis(analysis: str, require_surface: str | None = None) -> tuple[str | None, str | None]:
    for surface, tag in SEGMENT_RE.findall(analysis or ""):
        if tag.startswith("_"):
            continue
        if require_surface is not None and normalize_lemma(surface) != normalize_lemma(require_surface):
            continue
        if tag.startswith("/Adj"):
            return surface, "ADJ"
        if tag.startswith("/Adv"):
            return surface, "ADV"
        if tag.startswith("/Inj"):
            return surface, "INTJ"
        if tag.startswith("/Det"):
            return surface, "DET"
        if tag.startswith("/Num"):
            return surface, "NUM"
        if tag.startswith("/Post"):
            return surface, "ADP"
        if tag.startswith("/Prep"):
            return surface, "ADP"
        if tag.startswith("/Cnj"):
            return surface, "CCONJ"
        if tag.startswith("/Prev"):
            return surface, "PART"
        if tag.startswith("/QPtcl"):
            return surface, "PART"
        if tag.startswith("/N"):
            return surface, "NOUN"
        if tag.startswith("/V"):
            return surface, "VERB"
    return None, None


def normalize_lemma(value: Any) -> str:
    """Normalize lemma keys for matching CSV lemmas to emMorph surfaces."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _one_word_pipe_values(value: Any) -> list[str]:
    """Return one-word pipe-separated values in their original order."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = []
    for raw in str(value).split("|"):
        text = _strip_pos_fallback_context(raw)
        if text and " " not in text:
            parts.append(text)
    return list(dict.fromkeys(parts))


def _strip_pos_fallback_context(text: str) -> str:
    """
    Remove contextual gloss add-ons before one-word fallback checks.
    'Ackermaß 〈etwa 600 Quadratklafter〉' -> 'Ackermaß'
    """
    value = str(text).strip()
    value = re.sub(r"\s*[〈‹<][^〉›>]+[〉›>]\s*", "", value)
    value = re.sub(r"\s*\([^)]*\)\s*", "", value)
    return value.strip()


def _normalize_upos(value: str) -> str | None:
    """Map external POS labels to the reduced UPOS set used in this project."""
    if not value:
        return None
    tag = value.strip().upper()
    mapping = {
        "NOUN": "NOUN",
        "PROPN": "NOUN",
        "VERB": "VERB",
        "AUX": "VERB",
        "ADJ": "ADJ",
        "ADV": "ADV",
        "INTJ": "INTJ",
        "DET": "DET",
        "PRON": "DET",
        "NUM": "NUM",
        "ADP": "ADP",
        "CCONJ": "CCONJ",
        "SCONJ": "CCONJ",
        "PART": "PART",
        "CONJ": "CCONJ",
    }
    return mapping.get(tag)


def _combined_gloss_source(
    prefix: str,
    supporting_hun_glosses: list[str],
    supporting_ger_glosses: list[str],
) -> str:
    """Return a precise source label for combined gloss disambiguation."""
    if supporting_hun_glosses and supporting_ger_glosses:
        return f"{prefix}_combined_gloss"
    if supporting_hun_glosses:
        return f"{prefix}_hun_gloss"
    if supporting_ger_glosses:
        return f"{prefix}_ger_gloss"
    return f"{prefix}_combined_gloss"


def _split_pos_labels(value: str | None) -> list[str]:
    """Split a semicolon-joined POS label string into ordered unique labels."""
    if not value:
        return []
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    return list(dict.fromkeys(parts))


def _is_ambiguous_pos(value: str | None) -> bool:
    """Return True when a POS cell contains multiple candidate labels."""
    return len(_split_pos_labels(value)) > 1


def _has_homograph_value(value: Any) -> bool:
    """Return True when the row carries an explicit homograph number"""
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return text not in {"", "0", "nan", "NaN", "None"}


def _collect_fallback_pos_labels(
    glosses: list[str],
    gloss_to_pos: dict[str, str | None],
    *,
    candidates: set[str] | None = None,
    support_labels: set[str] | None = None,
) -> tuple[str | None, list[str]]:

    labels: set[str] = set()
    supporting_glosses: list[str] = []

    for gloss in glosses:
        gloss_labels = _split_pos_labels(gloss_to_pos.get(gloss) or "")
        if len(gloss_labels) > 1 and support_labels:
            pruned = [label for label in gloss_labels if label in support_labels]
            if pruned:
                gloss_labels = pruned
        if candidates is not None:
            gloss_labels = [label for label in gloss_labels if label in candidates]
        if not gloss_labels:
            continue
        labels.update(gloss_labels)
        supporting_glosses.append(gloss)

    if not labels:
        return None, []

    ordered = sorted(
        labels,
        key=lambda tag: UPOS_ORDER.index(tag) if tag in UPOS_ORDER else len(UPOS_ORDER),
    )
    return ";".join(ordered), supporting_glosses


def _merge_pos_label_strings(*values: str | None) -> str:
    """Merge semicolon-joined POS strings into one ordered unique label set."""
    labels: set[str] = set()
    for value in values:
        labels.update(_split_pos_labels(value))
    ordered = sorted(
        labels,
        key=lambda tag: UPOS_ORDER.index(tag) if tag in UPOS_ORDER else len(UPOS_ORDER),
    )
    return ";".join(ordered)


def _fallback_unambiguous_support_labels(
    *sources: tuple[list[str], dict[str, str | None]],
    candidates: set[str] | None = None,
) -> set[str]:

    supported: set[str] = set()
    for glosses, gloss_to_pos in sources:
        for gloss in glosses:
            labels = _split_pos_labels(gloss_to_pos.get(gloss) or "")
            if candidates is not None:
                labels = [label for label in labels if label in candidates]
            if len(labels) == 1:
                supported.add(labels[0])
    return supported

def make_spacy_german_pos_tagger(model: str = "de_core_news_sm") -> Callable[[list[str]], dict[str, str | None]]:

    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is not installed. Install spaCy and a German model such as "
            "'de_core_news_sm' first."
        ) from exc

    nlp = spacy.load(model, disable=["ner"])

    def tag_words(words: list[str]) -> dict[str, str | None]:
        tags: dict[str, str | None] = {}
        for doc in nlp.pipe(words, batch_size=256):
            if len(doc) != 1:
                tags[doc.text] = None
                continue
            tags[doc.text] = _normalize_upos(doc[0].pos_)
        return tags

    return tag_words


def move_columns_after(df: pd.DataFrame, columns: list[str], *, after: str) -> pd.DataFrame:
    """Move selected columns directly after another column."""
    ordered = [col for col in df.columns if col not in columns]
    insert_at = ordered.index(after) + 1 if after in ordered else len(ordered)
    for offset, column in enumerate(columns):
        if column in df.columns:
            ordered.insert(insert_at + offset, column)
    return df[ordered]


def normalize_integerish_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep year and certainty level metadata columns as plain strings, not floats like '1372.0'"""
    out = df.copy()
    for column in INTEGERISH_COLUMNS:
        if column not in out.columns:
            continue
        out[column] = out[column].map(_normalize_integerish_value)
    return out


def _normalize_integerish_value(value: Any) -> str:
    """Turn the year '1372.0' into '1372' while keeping blanks and non-integer text unchanged"""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _check_emmorph_paths(hfst_lookup: Path, hu_hfstol: Path) -> None:
    if not hfst_lookup.exists():
        raise FileNotFoundError(f"hfst-lookup not found: {hfst_lookup}")
    if not hu_hfstol.exists():
        raise FileNotFoundError(f"hu.hfstol not found: {hu_hfstol}")
