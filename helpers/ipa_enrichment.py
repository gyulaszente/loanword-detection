"""Add lemma IPA transcriptions to the compact UESz CSV export.
This is the reusable version of the phonemizer cell from `hu_corp.ipynb`.
It reads a CSV with a `lemma` column, phonemizes unique lemmas with eSpeak, and
writes an enriched CSV with `ipa` placed after `homograph`.
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Any
import platform
import pandas as pd


def _default_espeak_library() -> Path | None:
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
    if system == "Linux":
        return Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
    if system == "Darwin":
        return Path("/opt/homebrew/lib/libespeak-ng.dylib")
    return None

def _default_espeak_exe() -> Path | None:
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Program Files\eSpeak NG\espeak-ng.exe")
    if system == "Linux":
        return Path("/usr/bin/espeak-ng")
    if system == "Darwin":
        return Path("/opt/homebrew/bin/espeak-ng")
    return None

DEFAULT_ESPEAK_LIBRARY = _default_espeak_library()
DEFAULT_ESPEAK_EXE     = _default_espeak_exe()
INTEGERISH_COLUMNS = ["first_year", "certainty_level"]


def phonemize_lemmas(
    lemmas: list[str],
    *,
    espeak_library: str | Path | None = DEFAULT_ESPEAK_LIBRARY,
    espeak_exe: str | Path | None = DEFAULT_ESPEAK_EXE,
    language: str = "hu",
) -> dict[str, str]:
    """Return 'lemma -> ipa' for a list of lemmas using the local eSpeak setup."""
    normalized = sorted({
        normalize_lemma(value)
        for value in lemmas
        if normalize_lemma(value)
    })
    if not normalized:
        return {}

    try:
        _configure_espeak(espeak_library)
        from phonemizer import phonemize

        ipa_list = phonemize(
            normalized,
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=False,
        )
        return dict(zip(normalized, ipa_list))
    except ModuleNotFoundError:
        return _phonemize_lemmas_with_espeak_cli(
            normalized,
            espeak_exe=espeak_exe,
            language=language,
        )


def add_ipa_to_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    espeak_library: str | Path | None = DEFAULT_ESPEAK_LIBRARY,
    lemma_column: str = "lemma",
    language: str = "hu",
) -> pd.DataFrame:

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv, encoding="utf-8-sig", dtype=str).fillna("")
    if lemma_column not in df.columns:
        raise ValueError(f"Missing lemma column {lemma_column!r} in {input_csv}")

    _configure_espeak(espeak_library)
    lemmas = sorted({
        normalize_lemma(value)
        for value in df[lemma_column]
        if normalize_lemma(value)
    })
    print(f"Rows: {len(df)}")
    print(f"Unique lemmas: {len(lemmas)}")
    lemma_to_ipa = phonemize_lemmas(
        lemmas,
        espeak_library=espeak_library,
        language=language,
    )

    out = df.copy()
    out["ipa"] = out[lemma_column].map(lambda value: lemma_to_ipa.get(normalize_lemma(value), ""))

    out = normalize_integerish_columns(out)

    out = reorder_enrichment_columns(out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("Saved with IPA to:", output_csv)
    return out


def reorder_enrichment_columns(df: pd.DataFrame) -> pd.DataFrame:

    priority = [
        "lemma", "homograph", "ipa", "pos", "pos_source",
        "definition_hun", "translation", "stemtype",
        "corresponding_donor_word", "is_dialectal", "is_archaic", "is_obsolete",
        "first_year", "origin_type", "formation_type",
        "certainty_level", "certainty_label",
        "primary_donor_language", "primary_donor_family", "other_donor_languages",
        "entry_type", "entry_file",
    ]
    rest = [col for col in df.columns if col not in priority]
    ordered = [col for col in priority if col in df.columns] + rest
    return df[ordered]


def normalize_lemma(value: Any) -> str:
    """Normalize lemma keys for matching CSV lemmas to phonemizer output."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def move_columns_after(df: pd.DataFrame, columns: list[str], *, after: str) -> pd.DataFrame:
    """Move selected existing columns directly after another column."""
    ordered = [col for col in df.columns if col not in columns]
    insert_at = ordered.index(after) + 1 if after in ordered else len(ordered)
    for offset, column in enumerate(columns):
        if column in df.columns:
            ordered.insert(insert_at + offset, column)
    return df[ordered]


def normalize_integerish_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep integer-like metadata columns as plain strings, not floats like `1372.0`."""
    out = df.copy()
    for column in INTEGERISH_COLUMNS:
        if column not in out.columns:
            continue
        out[column] = out[column].map(_normalize_integerish_value)
    return out


def _normalize_integerish_value(value: Any) -> str:
    """Turn `1372.0` into `1372` while keeping blanks and non-integer text unchanged."""
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


def _configure_espeak(espeak_library: str | Path | None) -> None:
    """Point phonemizer at the local eSpeak NG DLL when a path is supplied."""
    if espeak_library is None:
        return
    espeak_library = Path(espeak_library)
    if not espeak_library.exists():
        raise FileNotFoundError(f"eSpeak NG library not found: {espeak_library}")

    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(str(espeak_library))


def _phonemize_lemmas_with_espeak_cli(
    lemmas: list[str],
    *,
    espeak_exe: str | Path | None = DEFAULT_ESPEAK_EXE,
    language: str = "hu",
) -> dict[str, str]:
    """Fallback IPA generation through the eSpeak NG executable."""
    if espeak_exe is None:
        raise FileNotFoundError("No eSpeak executable path supplied for CLI fallback.")

    exe_path = Path(espeak_exe)
    if not exe_path.exists():
        raise FileNotFoundError(f"eSpeak NG executable not found: {exe_path}")

    lemma_to_ipa: dict[str, str] = {}
    for lemma in lemmas:
        proc = subprocess.run(
            [str(exe_path), "-q", "--ipa", "-v", language, lemma],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "eSpeak NG CLI phonemization failed for "
                f"{lemma!r} with exit code {proc.returncode}: {proc.stderr.strip()}"
            )
        lemma_to_ipa[lemma] = _normalize_espeak_ipa(proc.stdout.strip())
    return lemma_to_ipa


def _normalize_espeak_ipa(text: str) -> str:
    """Normalize eSpeak IPA output to match the no-stress training data style."""
    return text.replace("ˈ", "").replace("ˌ", "").strip()
