"""
emmorph_test.py
Utility functions for working with emMorph morphological analyses.
"""

import re
import pandas as pd

# Compiled regexes

MORPHEME_RE = re.compile(r"([^][]+)\[[^]]+\]")
FIRST_POS_RE = re.compile(r"^[^[]+\[/([^]\|]+)")
_PAIR_RE = re.compile(r"([^][]+)\[([^]]+)\]")

# Low-level morpheme helpers

def extract_morphemes_from_analysis(analysis):
    """Return the list of surface morpheme strings from an emMorph analysis."""
    return MORPHEME_RE.findall(analysis)

"""
def extract_morpheme_code_pairs(analysis):
    #Return a list of (surface, code) tuples from an emMorph analysis string.
    return _PAIR_RE.findall(analysis or "")
"""

def extract_base_pos_from_analysis(analysis):
    """Extract the POS tag of the first (base) morpheme in an emMorph analysis."""
    match = FIRST_POS_RE.search(analysis or "")
    if not match:
        return None
    return match.group(1)


# POS normalisation and matching

_POS_MAP = {
    "NOUN":  "N",
    "PROPN": "N",
    "VERB":  "V",
    "AUX":   "V",
    "ADJ":   "Adj",
    "ADV":   "Adv",
    "INTJ":  "Interj",
}


def normalize_base_pos(pos):
    """
    Normalise a POS string to the short tags used by emMorph.  Returns a set of strings, or None.
    """
    if pd.isna(pos) or not pos:
        return None
    values = str(pos).split(";")
    return {_POS_MAP.get(v.strip(), v.strip()) for v in values}


def analysis_matches_base_pos(analysis, base_pos):

    expected = normalize_base_pos(base_pos)
    if not expected:
        return True
    actual = extract_base_pos_from_analysis(analysis)
    return actual in expected


# Stem extraction

def get_best_stem(row, analyses):
    """
    Return the stem from the best emMorph analysis for 'row'.
    """
    form  = row["derivation_form"]
    lemma = str(row["lemma"]).casefold()

    valid = [
        ana for ana in analyses.get(form, [])
        if "+?" not in ana.get("analysis", "")
        and ana.get("weight", float("inf")) < float("inf")
    ]
    if not valid:
        return None

    min_weight = min(ana["weight"] for ana in valid)
    candidates = [ana for ana in valid if ana["weight"] == min_weight]

    no_prev = [ana for ana in candidates if "[/Prev]" not in ana.get("analysis", "")]
    best = no_prev[0] if no_prev else candidates[0]

    pairs = _PAIR_RE.findall(best["analysis"])
    if not pairs:
        return None

    non_prev = [(s, c) for s, c in pairs if not c.startswith("/Prev")]
    if non_prev:
        return non_prev[0][0]

    return pairs[0][0]



def get_suffix_pairs_from_best_matching_analysis(row, analyses):

    lemma    = str(row["lemma"]).casefold()
    form     = row["derivation_form"]
    base_pos = row.get("base_pos")

    best_pairs = []

    for ana in analyses.get(form, []):
        analysis = ana.get("analysis", "")

        if "[Acc]" in analysis or "[Ess]" in analysis:
            continue
        if not analysis_matches_base_pos(analysis, base_pos):
            continue

        pairs = _PAIR_RE.findall(analysis)
        if not pairs:
            continue

        lexical_pairs = [(s, c) for s, c in pairs if not c.startswith("/Prev")]
        if not lexical_pairs:
            continue

        if lexical_pairs[0][0].casefold() != lemma:
            continue

        suffix_pairs = [(s, f"[{c}]") for s, c in lexical_pairs[1:]]
        if len(suffix_pairs) > len(best_pairs):
            best_pairs = suffix_pairs

    return best_pairs




# Suffix DataFrame helpers

def clean_suffix_df(df):

    return df[
        df["suffix_group"].notna()
        & (df["suffix_group"].astype(str) != "None")
        & (df["suffix_group"].astype(str).str.strip() != "")
        & df["emmorph_code"].notna()
        & ~df["emmorph_code"].astype(str).str.startswith("[/")
    ].copy()


def get_unstripped_suffix(row):

    lemma = str(row["lemma"]).lower()
    form  = str(row["derivation_form"]).lower()
    if form.startswith(lemma):
        return form[len(lemma):]
    return None


# Preverb analysis classification

def _is_lexical_pos(code):
    """True for tags like /N, /V, /Adj, /Adv, /Num — not /Prev, not derivational."""
    return code.startswith("/") and not code.startswith("/Prev")


def classify_preverb_analyses(row, analyses):

    lemma = str(row["lemma"]).casefold()
    form  = row["derivation_form"]

    has_prev_ana    = False
    has_no_prev_ana = False
    has_valid       = False
    best_preverb    = None
    best_stem       = None

    for ana in analyses.get(form, []):
        if ana.get("weight", float("inf")) == float("inf"):
            continue
        if "+?" in ana.get("analysis", ""):
            continue
        analysis = ana.get("analysis", "")

        if "[/Prev]" not in analysis:
            has_no_prev_ana = True
            continue

        has_prev_ana = True
        pairs    = _PAIR_RE.findall(analysis)
        non_prev = [(s, c) for s, c in pairs if not c.startswith("/Prev")]

        if not non_prev:
            continue

        stem_surface, stem_code = non_prev[0]

        # record first preverb/stem seen for examples
        if best_preverb is None:
            morphemes    = extract_morphemes_from_analysis(analysis)
            best_preverb = morphemes[0] if morphemes else None
            best_stem    = stem_surface

        if _is_lexical_pos(stem_code) and stem_surface.casefold() == lemma:
            has_valid    = True
            best_preverb = best_preverb  # already set
            best_stem    = stem_surface

    if not has_prev_ana:
        return {"category": "none", "preverb": None, "stem_after_prev": None}

    if has_valid:
        return {"category": "valid", "preverb": best_preverb, "stem_after_prev": best_stem}

    if (has_no_prev_ana
            and best_preverb is not None
            and lemma.startswith(best_preverb.casefold())):
        return {"category": "false_spurious", "preverb": best_preverb, "stem_after_prev": best_stem}

    return {"category": "false_atomic", "preverb": best_preverb, "stem_after_prev": best_stem}
