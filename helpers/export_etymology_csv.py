"""
Export the parsed UESz JSON entries to a compact CSV etymology table.
The JSON files remain the rich archival format. This exporter creates a flatter
database-like view: one row per single entry or family subentry, with the most
useful etymological metadata as columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_JSON_ROOT  = Path(__file__).parent.parent / "uesz_output" / "pdf_structured_json"
DEFAULT_OUTPUT_CSV = Path(__file__).parent.parent / "uesz_output" / "uesz_structured_db.csv"


CSV_COLUMNS = [
    "lemma",
    "homograph",
    "ipa",
    "pos",
    "pos_source",
    "definition_hun",
    "translation",
    "stemtype",
    "corresponding_donor_word",
    "is_dialectal",
    "is_archaic",
    "is_obsolete",
    "first_year",
    "first_year_is_generated",
    "first_year_generated_from",
    "first_year_range",
    "origin_type",
    "formation_type",
    "derivation_base",
    "derivation_base_origin",
    "compound_parts",
    "compound_parts_origins",
    "certainty_level",
    "certainty_label",
    "primary_donor_language",
    "primary_donor_family",
    "other_donor_languages",
    "entry_type",
    "entry_file",
]


STRUCTURAL_ENTRY_TYPES = {
    "prefix_family",
    "suffix_family",
    "roman_grouped_tilde_family",
}


def main() -> None:
    """Parse command-line arguments and write the CSV export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-root",
        type=Path,
        default=DEFAULT_JSON_ROOT,
        help="Root folder containing letter subfolders with parsed JSON entries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV file to write.",
    )
    args = parser.parse_args()

    row_count = write_csv(args.json_root, args.output)
    print(f"Wrote {row_count} rows to {args.output}")


def write_csv(json_root: Path, output_csv: Path) -> int:
    """Write the compact CSV export and return the number of rows written."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_csv.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in iter_csv_rows(json_root):
            writer.writerow(row)
            row_count += 1
    return row_count


def iter_csv_rows(json_root: Path) -> Iterable[dict[str, str]]:
    """Yield CSV-ready lexical row dicts for every parsed JSON file under json_root."""
    for json_path in sorted(json_root.rglob("*.json")):
        entry = json.loads(json_path.read_text(encoding="utf-8"))
        if is_duplicate_entry(entry):
            continue

        source_file = f"{json_path.parent.name}.txt"
        entry_file = json_path.name
        entry_type = str(entry.get("entry_type") or "")
        if entry_type in STRUCTURAL_ENTRY_TYPES:
            continue
        see_also = join_values(entry.get("see_also"))

        if "subentries" in entry and entry.get("subentries"):
            family_lemma = family_display_lemma(entry)
            base_indices = set(entry.get("base_of_word_family") or [])
            for subentry in entry["subentries"]:
                if is_duplicate_item(subentry):
                    continue
                row = make_row(
                    source_file=source_file,
                    entry_file=entry_file,
                    entry_type=entry_type,
                    family_lemma=family_lemma,
                    is_family_base=(subentry.get("subentry_index") in base_indices),
                    item=subentry,
                    see_also=see_also,
                )
                yield row
        else:
            if is_duplicate_item(entry):
                continue
            yield make_row(
                source_file=source_file,
                entry_file=entry_file,
                entry_type=entry_type,
                family_lemma="",
                is_family_base=None,
                item=entry,
                see_also=see_also,
            )


def is_duplicate_entry(entry: dict[str, Any]) -> bool:
    """Return True when the whole JSON entry is marked as a duplicate/same item."""
    return bool(
        entry.get("is_duplicate")
        or (entry.get("family_etymology") or {}).get("is_duplicate")
    )


def is_duplicate_item(item: dict[str, Any]) -> bool:
    """Return True when one row item should be omitted from the CSV as duplicate."""
    return bool(
        item.get("is_duplicate")
        or (item.get("subentry_etymology") or {}).get("is_duplicate")
    )


def _german_part(text: str) -> str:
    """Extract the German translation """
    if "|" not in text:
        return ""
    german = text.split("|", 1)[1]
    # Skip if the part after | starts with a grammatical category marker ‹ ›
    # which means | is separating POS categories, not Hungarian definition from German gloss
    stripped = german.strip()
    if stripped.startswith("‹"):
        return ""
    # Remove ÚESz markers: #, *, (↑), source references like (Nyr. 94: 479)
    german = re.sub(r"\s*#\s*$", "", german)
    german = re.sub(r"\s*\*\s*$", "", german)
    german = re.sub(r"\s*\(↑\)\s*$", "", german)
    german = re.sub(r"\s*\([^)]+\)\s*$", "", german)
    german = german.strip().strip("'").strip()
    return german


def _split_translation_options(value: str | list[str] | None) -> list[str]:
    """Return cleaned German translation options from structured fields."""
    if not value:
        return []
    if isinstance(value, list):
        raw_parts = value
    else:
        raw_parts = [value]

    parts: list[str] = []
    for raw in raw_parts:
        text = str(raw).strip()
        if not text:
            continue
        for part in text.split(";"):
            cleaned = part.strip()
            if cleaned:
                parts.append(cleaned)
    return parts


def _prefer_single_word_translations(translations: list[str]) -> list[str]:
    """
    Prefer one-word German translations.
    If at least one one-word option exists, multi-word options are dropped.
    Otherwise keep the original cleaned list.
    """
    cleaned = [t.strip() for t in translations if t and t.strip()]
    if not cleaned:
        return []
    single_word = [t for t in cleaned if " " not in t]
    chosen = single_word if single_word else cleaned
    return list(dict.fromkeys(chosen))


def extract_hungarian_definitions(item: dict[str, Any]) -> str:
    """Return all Hungarian definitions for an entry as a pipe-separated string.
    Uses same selection logic as the German translation extractor
    """
    meanings = item.get("meanings") or []
    definitions: list[str] = []
    for meaning in meanings:
        if not isinstance(meaning, dict):
            continue
        if meaning.get("form"):
            continue
        for attestation in meaning.get("attestations") or []:
            if not isinstance(attestation, dict):
                continue
            hun = attestation.get("definition_hun")
            if hun:
                definitions.extend(_split_translation_options(hun))
    return "|".join(_prefer_single_word_translations(definitions))


def extract_german_translations(item: dict[str, Any]) -> str:
    """Return all German translations for an entry as a pipe-separated string.
    Reads the structured 'translation_ger' field from attestation objects.
    Falls back to parsing the 'text' field for entries parsed before
    the structured meaning format was introduced.
    """
    meanings = item.get("meanings") or []
    translations: list[str] = []
    for meaning in meanings:
        if not isinstance(meaning, dict):
            continue

        if meaning.get("form"):
            continue

        for attestation in meaning.get("attestations") or []:
            if not isinstance(attestation, dict):
                continue
            ger = attestation.get("translation_ger")
            if ger:
                translations.extend(_split_translation_options(ger))

        if not meaning.get("attestations"):
            raw = meaning.get("text")
            if raw:
                german = _german_part(str(raw))
                if german:
                    translations.extend(_split_translation_options(german))
    return "|".join(_prefer_single_word_translations(translations))


def make_row(
    *,
    source_file: str,
    entry_file: str,
    entry_type: str,
    family_lemma: str,
    is_family_base: bool | None,
    item: dict[str, Any],
    see_also: str,
) -> dict[str, str]:
    """Build one flat CSV row from a single entry or one family subentry."""
    flags = item.get("flags") or {}

    return {
        "entry_file": entry_file,
        "entry_type": entry_type,
        "lemma": text_value(item.get("lemma")),
        "homograph": text_value(item.get("homograph")),
        "ipa": "",           # filled later by ipa_enrichment.py
        "pos": text_value(item.get("pos")),
        "pos_source": text_value(item.get("pos_source")),
        "definition_hun": extract_hungarian_definitions(item),
        "translation": extract_german_translations(item),
        "stemtype": text_value(item.get("stemtype")),
        "corresponding_donor_word": text_value(item.get("corresponding_donor_word")),
        "is_dialectal": bool_to_yes_no(flags.get("is_dialectal", False)),
        "is_archaic": bool_to_yes_no(flags.get("is_archaic", False)),
        "is_obsolete": bool_to_yes_no(flags.get("is_obsolete", False)),
        "first_year": integerish_value(item.get("first_year_of_attestation")),
        "first_year_is_generated": bool_to_yes_no(item.get("first_year_of_attestation_is_generated", False)),
        "first_year_generated_from": text_value(item.get("first_year_of_attestation_generated_from")),
        "first_year_range": join_values(item.get("first_year_of_attestation_range")),
        "origin_type": text_value(item.get("origin_type")),
        "formation_type": formation_type_value(item.get("formation_type")),
        "derivation_base": text_value(item.get("derivation_base")),
        "derivation_base_origin": text_value(item.get("derivation_base_origin")),
        "compound_parts": "|".join(item.get("compound_parts") or []),
        "compound_parts_origins": "|".join(
            (item.get("compound_parts_origins") or {}).get(p, "")
            for p in (item.get("compound_parts") or [])
        ),
        "certainty_level": integerish_value(item.get("origin_certainty_level")),
        "certainty_label": text_value(item.get("certainty_label")),
        "primary_donor_language": text_value(item.get("primary_donor_language")),
        "primary_donor_family": text_value(item.get("primary_donor_family")),
        "other_donor_languages": join_values(item.get("other_donor_languages")),
    }


def formation_type_value(value: Any) -> str:
    """Return a non-empty formation_type value for compact exports."""
    return text_value(value) or "other"


def family_display_lemma(entry: dict[str, Any]) -> str:
    """Return the family label to repeat on subentry rows."""
    if entry.get("lemma"):
        return text_value(entry["lemma"])
    head_lines = entry.get("head_lines") or []
    if head_lines:
        return text_value(head_lines[0])
    return ""


def first_attestation_source(attestations: Any, first_year: Any) -> str:
    """
    Return the source attached to the selected first attestation year.
    Antedated/forged dates marked as excluded are ignored
    """
    if not isinstance(attestations, list):
        return ""

    for attestation in attestations:
        if not isinstance(attestation, dict):
            continue
        if attestation.get("date_type") == "excluded_antedated":
            continue
        if attestation.get("year") == first_year:
            return text_value(attestation.get("source"))
    return ""


def has_nyelvujitas_note(item: dict[str, Any]) -> bool:
    """Detect whether a subentry etymology contains a nyelvújítás note block."""
    etymology = item.get("subentry_etymology") or {}
    block_groups = [
        etymology.get("blocks") or [],
        etymology.get("shared_blocks") or [],
    ]
    return any(
        isinstance(block, dict) and block.get("role") == "nyelvujitas"
        for blocks in block_groups
        for block in blocks
    )


def bool_to_yes_no(value: Any) -> str:
    """Render booleans as yes/no while leaving not-applicable values blank."""
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return "yes"
        if normalized in {"false", "no", "0"}:
            return "no"
    return "yes" if bool(value) else "no"


def join_values(value: Any) -> str:
    """Join list-like JSON values into a stable semicolon-separated CSV cell."""
    if not value:
        return ""
    if isinstance(value, list):
        return "; ".join(text_value(item) for item in value if text_value(item))
    return text_value(value)


def text_value(value: Any) -> str:
    """Convert scalar JSON values to CSV text without Python None/True/False spellings."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return bool_to_yes_no(value)
    return str(value)


def integerish_value(value: Any) -> str:
    """Render integer-like metadata values without a trailing decimal part."""
    text = text_value(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


if __name__ == "__main__":
    main()
