"""
Text-cleaning helper functions for the ÚESz extraction pipeline.
The parser needs the original HTML-like markup, but many later steps
need plain text for matching, filenames, and JSON output. This module keeps those
cleaning rules in one place, including abbreviation normalization.
"""

import re
from . import ueszweb_abbreviations as ab
from . import regexes as rx
from typing import Any, Optional


TAG_RE = re.compile(r"</?[bi]>")

def strip_style_tags(text: str) -> str:
    """Remove only the lightweight <b>/<i> style tags used in the TXT source."""
    if not text:
        return text
    return TAG_RE.sub("", text)

def normalize_plain_text(text: str) -> str:
    """
    Remove style tags and normalize spacing.
    Keep the linguistic symbols/signs themselves.
    """
    if not text:
        return text
    text = strip_style_tags(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_entry_lines(raw_entry_text: str) -> list[dict[str, str]]:
    """
    Keep raw ONLY temporarily for parsing,
    but mark it clearly as internal.
    """
    result = []
    for ln in raw_entry_text.splitlines():
        if not ln.strip():
            continue

        raw = ln.rstrip()
        plain = normalize_plain_text(raw)

        result.append({
            "raw": raw,      # <- internal use only
            "plain": plain
        })

    return result

def normalize_filename(text: str) -> str:
    """Convert a lemma/homograph label into a filesystem-safe JSON filename part."""
    if not text:
        return "unknown"

    text = text.lower()

    # keep accents, just clean unsafe chars
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text or "unknown"

# Build a set of tokens that should *not* be normalized by normalize_surface_token.
# 1) keys from the HUNGARIAN_ABBREVIATIONS_DICT
PROTECTED_TOKENS: set[str] = set(ab.HUNGARIAN_ABBREVIATIONS_DICT.keys())

# 2) keys from the sign dictionaries (only those that behave like standalone tokens)
for d in (
    ab.UESZ_SIGNS_ANYWHERE,
    ab.UESZ_SIGNS_HEADWORD,
    ab.UESZ_SIGNS_HISTORY_YEARS,
    ab.UESZ_SIGNS_HISTORY_DATA,
    ab.UESZ_SIGNS_MEANING,
    ab.UESZ_SIGNS_ETYMOLOGY,
    ab.UESZ_SIGNS_ETYMOLOGY_BLOCKS,
    ab.UESZ_SIGNS_REFERENCES,
):
    PROTECTED_TOKENS.update(d.keys())

# 3) all explicit abbreviation forms in ABBREVIATIONS_DICT values
for forms in ab.ABBREVIATIONS_DICT.values():
    PROTECTED_TOKENS.update(forms)


def normalize_surface_token(tok: str, lowercase: bool = True) -> str:

    tok = tok.strip()
    if not tok:
        return ""

    if tok in PROTECTED_TOKENS:
        return tok

    if lowercase:
        tok = tok.lower()

    has_final_dot = tok.rstrip(",:;!?").endswith(".")

    tok = re.sub(r"^[^\w(]+", "", tok, flags=re.UNICODE)

    tok = re.sub(r"[^\w).]+$", "", tok, flags=re.UNICODE)

    if not tok:
        return ""
    tok = re.sub(r"[,:;!?]+$", "", tok)

    if not tok:
        return ""

    # Normalize final-dot behavior
    if has_final_dot:
        tok = tok.rstrip(".") + "."
    else:
        tok = tok.rstrip(".")

    return tok


def get_entry_file_base(entry: dict[str, Any]) -> tuple[str, str]:
    """
    Return (lemma, homograph_string) for file naming.
    """
    if entry.get("entry_type") in {
        "prefix_family",
        "roman_grouped_tilde_family",
        "suffix_family",
    }:
        lemma = entry.get("lemma") or "unknown"
        hom = entry.get("homograph")
        return lemma, str(hom) if hom is not None else ""

    if entry.get("entry_type") == "single":
        lemma = entry.get("lemma") or "unknown"
        hom = entry.get("homograph")
        return lemma, str(hom) if hom is not None else ""

    subentries = entry.get("subentries", [])
    if subentries:
        first = subentries[0]
        lemma = first.get("lemma") or "unknown"
        hom = first.get("homograph")
        return lemma, str(hom) if hom is not None else ""

    return "unknown", ""
