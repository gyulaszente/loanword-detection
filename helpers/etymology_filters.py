"""Small helpers for inspecting the compact etymology CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CSV_PATH = Path(__file__).with_name("etymology_db_current.csv")


def load_etymology_csv(path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the compact CSV and normalize its simple yes/no and numeric fields."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return prepare_etymology_df(df)


def prepare_etymology_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize current CSV columns for filtering and notebook inspection."""
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()

    for col in ["first_year", "certainty_level"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["is_dialectal", "is_archaic", "is_obsolete"]:
        if col in out.columns:
            out[col] = out[col].map(yes_no_to_bool)
        else:
            out[col] = False

    for col in [
        "lemma",
        "ipa",
        "pos",
        "origin_type",
        "formation_type",
        "primary_donor_family",
        "primary_donor_language",
        "entry_type",
        "entry_file",
    ]:
        if col not in out.columns:
            out[col] = ""

    return out


def yes_no_to_bool(value) -> bool:
    """Convert yes/no, true/false, and 1/0 cells to booleans."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return True
    if text in {"no", "false", "0", ""}:
        return False
    return bool(value)


def available_values(df: pd.DataFrame, column: str, include_missing: bool = False) -> list[str]:
    """Return sorted distinct values for a classification column."""
    values = df[column] if column in df.columns else pd.Series(dtype="object")
    cleaned = values.fillna("").astype(str).str.strip()
    if not include_missing:
        cleaned = cleaned[cleaned != ""]
    return sorted(cleaned.unique().tolist())


def value_counts_table(
    df: pd.DataFrame,
    column: str,
    *,
    include_missing: bool = True,
    missing_label: str = "Missing",
) -> pd.DataFrame:
    """Return a compact frequency table for one classification column."""
    values = df[column] if column in df.columns else pd.Series(dtype="object")
    cleaned = values.fillna("").astype(str).str.strip()
    if include_missing:
        cleaned = cleaned.replace("", missing_label)
    else:
        cleaned = cleaned[cleaned != ""]
    return (
        cleaned.value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )


def filter_entries(
    df: pd.DataFrame,
    *,
    origin_types: str | Iterable[str] | None = None,
    formation_types: str | Iterable[str] | None = None,
    donor_families: str | Iterable[str] | None = None,
    donor_languages: str | Iterable[str] | None = None,
    include_dialectal: bool = True,
    include_archaic: bool = True,
    include_obsolete: bool = True,
    year_min: int | None = None,
    year_max: int | None = None,
    lemma_contains: str | None = None,
) -> pd.DataFrame:
    """Filter entries by origin, formation, donor metadata, flags, and year."""
    out = df.copy()
    mask = pd.Series(True, index=out.index)

    mask &= _isin_if_given(out, "origin_type", origin_types)
    mask &= _isin_if_given(out, "formation_type", formation_types)
    mask &= _isin_if_given(out, "primary_donor_family", donor_families)
    mask &= _isin_if_given(out, "primary_donor_language", donor_languages)

    if not include_dialectal:
        mask &= ~out.get("is_dialectal", pd.Series(False, index=out.index)).astype(bool)

    if not include_archaic:
        mask &= ~out.get("is_archaic", pd.Series(False, index=out.index)).astype(bool)

    if not include_obsolete:
        mask &= ~out.get("is_obsolete", pd.Series(False, index=out.index)).astype(bool)


    if year_min is not None and "first_year" in out.columns:
        mask &= out["first_year"].ge(year_min)
    if year_max is not None and "first_year" in out.columns:
        mask &= out["first_year"].le(year_max)

    if lemma_contains:
        mask &= out["lemma"].fillna("").astype(str).str.contains(
            lemma_contains,
            case=False,
            regex=False,
        )

    return out[mask].copy()


def lemmas_for(
    df: pd.DataFrame,
    *,
    origin_type: str | None = None,
    formation_type: str | None = None,
    donor_family: str | None = None,
    donor_language: str | None = None,
    include_dialectal: bool = True,
    include_archaic: bool = True,
    include_obsolete: bool = True,
    columns: Iterable[str] = (
        "lemma",
        "homograph",
        "first_year",
        "origin_type",
        "formation_type",
        "primary_donor_language",
        "primary_donor_family",
        "entry_file",
    ),
) -> pd.DataFrame:
    """Return rows/lemmas belonging to one specific type or donor group."""
    result = filter_entries(
        df,
        origin_types=origin_type,
        formation_types=formation_type,
        donor_families=donor_family,
        donor_languages=donor_language,
        include_dialectal=include_dialectal,
        include_archaic=include_archaic,
        include_obsolete=include_obsolete,
    )
    cols = [col for col in columns if col in result.columns]
    return result[cols].sort_values(["lemma", "homograph"], na_position="last")


def lemmas_grouped_by(df: pd.DataFrame, column: str) -> dict[str, list[str]]:
    """Return {category: sorted unique lemmas} for a classification column."""
    if column not in df.columns:
        return {}
    grouped: dict[str, list[str]] = {}
    tmp = df.copy()
    tmp[column] = tmp[column].fillna("").astype(str).str.strip().replace("", "Missing")
    for value, group in tmp.groupby(column):
        grouped[value] = sorted(group["lemma"].dropna().astype(str).unique().tolist())
    return grouped


def origin_formation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate origin_type by formation_type."""
    origin = df["origin_type"].fillna("").astype(str).str.strip().replace("", "Missing")
    formation = df["formation_type"].fillna("").astype(str).str.strip().replace("", "Missing")
    return pd.crosstab(origin, formation, margins=True)


def donor_family_origin_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate donor family by origin type."""
    family = df["primary_donor_family"].fillna("").astype(str).str.strip().replace("", "Missing")
    origin = df["origin_type"].fillna("").astype(str).str.strip().replace("", "Missing")
    return pd.crosstab(family, origin, margins=True)


def _isin_if_given(
    df: pd.DataFrame,
    column: str,
    values: str | Iterable[str] | None,
) -> pd.Series:
    if values is None:
        return pd.Series(True, index=df.index)
    wanted = _as_set(values)
    if column in df.columns:
        cleaned = df[column].fillna("").astype(str).str.strip()
    else:
        cleaned = pd.Series("", index=df.index)
    return cleaned.isin(wanted)


def _as_set(values: str | Iterable[str]) -> set[str]:
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}
