"""
phonology.py

"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

# Constants

# ASCII escapes
IPA_LENGTH = "\u02d0"
IPA_SMALL_CAP_I = "\u026a"
IPA_A = "\u0251"
IPA_EPSILON = "\u025b"
IPA_OPEN_O = "\u0254"
IPA_O_SLASH = "\u00f8"

IK_SUFFIX_IPA_RE = re.compile(f"(?:i|{IPA_SMALL_CAP_I})k$")

"""Vowel-harmony class mapping for the Hungarian IPA inventory used here.
B = back, 
F = front unrounded, 
R = front rounded"""
VOWEL_CLASS: dict[str, str] = {
    IPA_A: "B", "a": "B", IPA_OPEN_O: "B", "o": "B", "u": "B",
    "e": "F", "i": "F", IPA_EPSILON: "F",
    IPA_O_SLASH: "R", "y": "R",
}

VOWEL_BASES: set[str] = set(VOWEL_CLASS.keys())

V_PREFIXES: tuple[str, ...] = (
    "vcount_",
    "vswitch_",
    "v_harmonic",
    "v_mixed",
    "v_disharmonic",
    "v_has_round",
    "v_no_round",
    "vstr_",
    "vstrlen_")



def yes_no_to_bool(value) -> bool:
    """Normalise CSV yes/no/True/False/0/1 flags to Python bool."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return True
    if text in {"no", "false", "0", ""}:
        return False
    return bool(value)


# IPA tokenisation

def strip_ipa_ik_suffix(ipa: str) -> str:

    if not isinstance(ipa, str):
        return ipa
    return IK_SUFFIX_IPA_RE.sub("", ipa.strip())


def tokenize_ipa_transcribed_word(ipa: str) -> list[str]:

    if not isinstance(ipa, str):
        return []

    affricates: set[str] = {"ts", "dz", "t\u0283", "d\u0292"}

    tokens: list[str] = []
    i = 0
    while i < len(ipa):

        if i + 1 < len(ipa) and ipa[i: i + 2] in affricates:
            ch2 = ipa[i: i + 2]
            if i + 2 < len(ipa) and ipa[i + 2] == IPA_LENGTH:
                tokens.append(ch2 + IPA_LENGTH)
                i += 3
            else:
                tokens.append(ch2)
                i += 2
            continue

        ch = ipa[i]

        if i + 1 < len(ipa) and ipa[i + 1] == IPA_LENGTH:
            tokens.append(ch + IPA_LENGTH)
            i += 2
            continue

        tokens.append(ch)
        i += 1

    return tokens


# n-gram extraction

def phoneme_ngrams(ipa: str, n: int,
    skip_initial_unbounded: bool = True,
    strip_final_ik: bool = True) -> list[str]:

    if strip_final_ik:
        ipa = strip_ipa_ik_suffix(ipa)

    tokens = tokenize_ipa_transcribed_word(ipa)
    if not tokens:
        return []

    tokens = ["^"] + tokens + ["$"]
    result: list[str] = []
    length = len(tokens)

    for start in range(length):
        if skip_initial_unbounded and start == 1:
            continue

        core_count = 0
        end = start

        while end < length and core_count < n:
            if tokens[end] not in {"^", "$"}:
                core_count += 1
            end += 1

        if core_count == n:

            if end < length and tokens[end] == "$":
                end += 1
            result.append("".join(tokens[start:end]))
        else:
            break

    return result


def collect_ngrams_multi(ipa: str, ns: tuple[int, ...] = (2, 3, 4),
    skip_initial_unbounded: bool = True,
    strip_final_ik: bool = True,) -> list[str]:

    grams: list[str] = []

    if not isinstance(ipa, str):
        return grams

    for n in ns:
        for g in phoneme_ngrams(ipa, n=n,
            skip_initial_unbounded=skip_initial_unbounded,
            strip_final_ik=strip_final_ik):
            grams.append(g)     # keep ^sp, ra$
            if g.startswith("^"):
                grams.append(g[1:])   # also keep the same n-gram without the word-start marker
            elif g.endswith("$"):
                grams.append(g[:-1])    # also keep the same n-gram without the word-start marker

    return grams


def _normalize_pipe_values(value) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip().lower()
    if not text:
        return set()
    return {part.strip() for part in text.split("|") if part.strip()}


# Vowel harmony features

def vowel_string(ipa: str, keep_length: bool = False) -> str:

    if not isinstance(ipa, str):
        return ""

    out: list[str] = []
    for t in tokenize_ipa_transcribed_word(ipa):
        base = t[0]
        if base in VOWEL_BASES:
            if keep_length and t.endswith(IPA_LENGTH):
                out.append(base + IPA_LENGTH)
            else:
                out.append(base)

    return "".join(out)


def _vowel_count_from_vstr(vstr: str) -> int:
    """Count vowels in a vowel-string; length marks are not counted."""
    return sum(1 for ch in vstr if ch != IPA_LENGTH) if vstr else 0


def vowel_pattern_tokens(ipa: str) -> list[str]:

    if not isinstance(ipa, str):
        return ["V_NONE"]

    vstr = vowel_string(ipa, keep_length=False)
    vcount = _vowel_count_from_vstr(vstr)

    tokens: list[str] = []

    # --- vowel count ---
    if vcount == 0:
        return ["VCOUNT_0"]
    elif vcount == 1:
        tokens.append("VCOUNT_1")
    elif vcount == 2:
        tokens.append("VCOUNT_2")
    elif vcount == 3:
        tokens.append("VCOUNT_3")
    else:
        tokens.append("VCOUNT_4PLUS")

    classes = [VOWEL_CLASS.get(ch, "?") for ch in vstr if ch != IPA_LENGTH]
    uniq = set(classes)

    switches = sum(classes[i] != classes[i - 1] for i in range(1, len(classes)))
    if switches == 1:
        tokens.append("VSWITCH_1")
    elif switches > 1:
        tokens.append("VSWITCH_2PLUS")

    tokens.append("V_HARMONIC" if switches == 0 else "V_MIXED")

    tokens.append("V_HAS_ROUND" if "R" in uniq else "V_NO_ROUND")

    if "B" in uniq and ("F" in uniq or "R" in uniq):
        tokens.append("V_DISHARMONIC")

    return tokens


def extract_vowel_phonemes(ipa: str) -> list[str]:
    """Return vowel base phonemes in word order (long vowels count as one)."""
    if not isinstance(ipa, str):
        return []

    return [t[0]
        for t in tokenize_ipa_transcribed_word(ipa)
        if t[0] in VOWEL_BASES]


def vowel_string_token(ipa: str, keep_length: bool = False, max_len: int = 6) -> str:

    vs = vowel_string(ipa, keep_length=keep_length)
    vcount = _vowel_count_from_vstr(vs)

    if vcount == 0:
        return "VSTR_NONE"

    if len(vs) > max_len:
        vs = vs[:max_len] + "_TRUNC"
    return f"VSTR_{vs}"


def vowel_string_length_token(ipa: str, max_len: int = 8) -> str:

    vs = vowel_string(ipa, keep_length=True)
    vcount = _vowel_count_from_vstr(vs)

    if vcount == 0:
        return "VSTRLEN_NONE"

    if len(vs) > max_len:
        vs = vs[:max_len] + "_TRUNC"
    return f"VSTRLEN_{vs}"


# Document builder

def make_ngram_documents(
    df: pd.DataFrame,
    ipa_col: str = "ipa",
    ns: tuple[int, ...] = (2, 3, 4),
    add_vowel_string: bool = True,
    add_vowel_patterns: bool = True,
    add_vowel_length_string: bool = True,
    skip_initial_unbounded: bool = False,
    strip_final_ik: bool = True,
    dedupe_inherited_family_ngrams: bool = True,
    origin_col: str = "origin_type",
    entry_type_col: str = "entry_type",
    family_key_col: str = "entry_file",
    year_col: str = "first_year") -> pd.Series:


    def ngram_tokens(ipa: str) -> list[str]:
        return collect_ngrams_multi(
            ipa=ipa,
            ns=ns,
            skip_initial_unbounded=skip_initial_unbounded,
            strip_final_ik=strip_final_ik)

    def vowel_tokens(ipa: str) -> list[str]:
        tokens: list[str] = []
        if add_vowel_string:
            tokens.append(vowel_string_token(ipa))
        if add_vowel_length_string:
            tokens.append(vowel_string_length_token(ipa))
        if add_vowel_patterns:
            tokens.extend(vowel_pattern_tokens(ipa))
        return tokens

    docs = pd.Series(index=df.index, dtype="object")

    seen_ngram_tokens_by_family: dict[str, set[str]] = {}

    year_for_sort = pd.to_numeric(df.get(year_col, pd.Series(index=df.index, dtype="object")), errors="coerce")
    order_df = df.assign(_year_sort=year_for_sort).copy()

    for idx, row in order_df.sort_values("_year_sort", na_position="last").iterrows():
        ipa = row.get(ipa_col)
        grams = ngram_tokens(ipa)
        inherited_parts = _normalize_pipe_values(row.get(origin_col))
        is_inherited = "inherited" in inherited_parts
        is_family_member = str(row.get(entry_type_col) or "").strip().lower() == "family"
        family_key = str(row.get(family_key_col) or "").strip()

        if dedupe_inherited_family_ngrams and is_inherited and is_family_member and family_key:
            seen = seen_ngram_tokens_by_family.setdefault(family_key, set())
            unique_grams = [g for g in grams if g not in seen]
            seen.update(grams)
            grams = unique_grams

        tokens = grams + vowel_tokens(ipa)
        docs.at[idx] = " ".join(tokens)

    return docs


# Entry filtering
def filter_real_entries(
    df: pd.DataFrame,
    ref_col: str = "reference_to_lemma",
    ipa_col: str = "ipa",
    min_ipa_len: int = 3,
    exclude_old_words: bool = False,
    old_word_cols: tuple[str, ...] = ("is_dialectal", "is_archaic", "is_obsolete"),
    exclude_formation_types: tuple[str, ...] = (
        "compound", "phrasal_compound"),
    formation_col: str = "formation_type",
    origin_col: str = "origin_type",
    entry_type_col: str = "entry_type",
    require_family_origin_and_formation: bool = True,
    min_first_year: int | None = None,
    year_col: str = "first_year") -> pd.DataFrame:

    sub = df.copy()

    # old-word exclusion
    if exclude_old_words:
        for c in old_word_cols:
            if c in sub.columns:
                sub[c] = sub[c].map(yes_no_to_bool)
            else:
                sub[c] = False

        old_mask = False
        for c in old_word_cols:
            old_mask = old_mask | sub[c]

        sub = sub[~old_mask].copy()

    # cross-references
    if ref_col in sub.columns:
        ref_empty = sub[ref_col].isna() | (sub[ref_col].astype(str).str.strip() == "")
        sub = sub[ref_empty].copy()

    # IPA present
    sub = sub[sub[ipa_col].notna()].copy()

    # IPA length
    sub = sub[sub[ipa_col].astype(str).str.strip().str.len() >= min_ipa_len].copy()

    # formation-type exclusion
    if exclude_formation_types and formation_col in sub.columns:
        excl = {t.lower().strip() for t in exclude_formation_types}
        formation_clean = sub[formation_col].astype(str).str.strip().str.lower()
        n_before = len(sub)
        formation_parts = formation_clean.apply(_normalize_pipe_values)
        exclude_mask = formation_parts.apply(lambda parts: bool(parts & excl))
        sub = sub[~exclude_mask].copy()
        print(
            f"Excluded {n_before - len(sub)} rows "
            f"with formation_type containing {sorted(excl)}"
        )

    # family subentries need explicit recovered labels
    if (require_family_origin_and_formation
        and entry_type_col in sub.columns
        and origin_col in sub.columns
        and formation_col in sub.columns):
        entry_type_clean = sub[entry_type_col].astype(str).str.strip().str.lower()
        origin_empty = sub[origin_col].isna() | (sub[origin_col].astype(str).str.strip() == "")
        formation_empty = sub[formation_col].isna() | (sub[formation_col].astype(str).str.strip() == "")
        missing_family_label = entry_type_clean.eq("family") & (origin_empty | formation_empty)
        n_before = len(sub)
        sub = sub[~missing_family_label].copy()
        if len(sub) < n_before:
            print(f"Excluded {n_before - len(sub)} family rows "
                "without recovered origin_type or formation_type")

    if "lemma" in sub.columns:
        year_for_sort = pd.to_numeric(sub.get(year_col, pd.Series(dtype=float)), errors="coerce")
        sub = sub.assign(_year_sort=year_for_sort)
        dedupe_cols = ["lemma"]
        for col in ("homograph", "pos"):
            if col in sub.columns:
                dedupe_cols.append(col)
        n_before = len(sub)
        sub = (sub.sort_values("_year_sort", na_position="last")
                  .drop_duplicates(subset=dedupe_cols, keep="first")
                  .drop(columns="_year_sort")
                  .copy())
        if len(sub) < n_before:
            print(
                f"Deduplicated {n_before - len(sub)} rows "
                f"(same {', '.join(dedupe_cols)}, kept earliest first_year)"
            )

    if min_first_year is not None and year_col in sub.columns:
        year = pd.to_numeric(sub[year_col], errors="coerce")
        n_before = len(sub)
        sub = sub[year >= min_first_year].copy()
        print(
            f"Excluded {n_before - len(sub)} rows "
            f"with {year_col} < {min_first_year} (or missing year)"
        )

    return sub


# Labeling
def add_custom_labels(
    df: pd.DataFrame,
    family_col: str = "primary_donor_family",
    donor_language_col: str = "primary_donor_language",
    origin_col: str = "origin_type",
    formation_col: str = "formation_type",
    entry_type_col: str = "entry_type",
    treat_unknown_as_nonloan: bool = False) -> pd.DataFrame:

    out = df.copy()

    def _clean_str_col(col_name: str, lower: bool = False) -> pd.Series:
        s = out.get(col_name, pd.Series(index=out.index, dtype="object"))
        s = s.where(s.notna(), None)
        mask = s.notna()
        if mask.any():
            s = s.copy()
            s[mask] = s[mask].astype(str).str.strip()
            if lower:
                s[mask] = s[mask].str.lower()
        return s

    out["family_clean"] = _clean_str_col(family_col, lower=False)
    out["donor_language_clean"] = _clean_str_col(donor_language_col, lower=False)
    out["origin_clean"] = _clean_str_col(origin_col, lower=True)
    out["formation_clean"] = _clean_str_col(formation_col, lower=True)
    out["entry_type_clean"] = _clean_str_col(entry_type_col, lower=True)

    # Origins excluded entirely from classification
    excluded_origins: set[str] = {"calque", "loanblend"}

    loan_like_origin: set[str] = {"loan", "international", "wanderword"}
    native_like_origin: set[str] = {
        "inherited",
        "internal",
        "onomatopoetic",
        "fictional",
    }

    def decide(row) -> bool | None:
        fam = row["family_clean"]
        donor_lang = row["donor_language_clean"]
        ori = row["origin_clean"]

        if isinstance(ori, str) and "|" in ori:
            parts = {p.strip() for p in ori.split("|")}
            if parts & excluded_origins:
                return None
            if parts <= loan_like_origin:
                return True
            if parts <= native_like_origin:
                return False
            return None

        if isinstance(ori, str) and ori in excluded_origins:
            return None

        if isinstance(ori, str) and ori in loan_like_origin:
            return True
        if isinstance(ori, str) and ori in native_like_origin:
            return False

        fam_missing = fam is None or pd.isna(fam) or str(fam).strip() == ""
        fam_is_uralic = (not fam_missing) and str(fam).strip() == "Uralic"

        if treat_unknown_as_nonloan and ori == "unknown" and (fam_missing or fam_is_uralic):
            return False
        return None

    out["is_loan_custom"] = out.apply(decide, axis=1)
    return out

"""
Dataset builders, included Mojibake characters just in case """
_DONOR_LANGUAGE_TO_TARGET_FAMILY_RULES: list[tuple[str, str]] = [
    (r"english", "English"),
    (r"angol", "English"),
    (r"german|yiddish", "German"),
    (r"nÃ©met|jiddis", "German"),
    (r"latin|greek", "Latin"),
    (r"latin|gÃ¶rÃ¶g", "Latin"),
    (r"slavic|russian|ukrainian|serbian|croatian|serbo-croatian|slovak|slovene|"
    r"czech|polish|bulgarian|macedonian|belarusian|church slavonic|old church slavonic|",
    "Slavic"),
    (r"szlÃ¡v|orosz|ukrÃ¡n|szerb|horvÃ¡t|szerbhorvÃ¡t|szlovÃ¡k|szlovÃ©n|cseh|lengyel|"
    r"bolgÃ¡r|macedÃ³n|ruszin|belarusz|Ã³egyhÃ¡zi szlÃ¡v|Ã³szlÃ¡v",
    "Slavic"),
    (r"turkish|ottoman|chagatai|kipchak|cuman|tatar|azeri|azerbaijani|kazakh|"
    r"kyrgyz|turkmen|uyghur|bashkir|chuvash",
    "Turkic"),
    (r"tÃ¶rÃ¶k|oszmÃ¡n|csagatÃ¡j|kipcsak|kun|tatÃ¡r|azeri|azerbajdzsÃ¡ni|kazah|"
    r"kirgiz|tÃ¼rkmÃ©n|ujgur|baskÃ­r|csuvas",
    "Turkic"),
    (r"french|italian|spanish|portuguese|romanian|provenÃ§al|occitan|catalan|"
    r"venetian|neapolitan|sardinian|dalmatian",
    "Romance"),
    (r"francia|olasz|spanyol|portugÃ¡l|romÃ¡n|provanszÃ¡l|okcitÃ¡n|katalÃ¡n|"
    r"velencei|nÃ¡polyi|szardÃ­niai|dalmÃ¡t",
    "Romance"),]


def _map_donor_language_to_target_family(language: str | None) -> str | None:
    if language is None or pd.isna(language):
        return None
    lang = str(language).strip().lower()
    if not lang:
        return None
    for pattern, family in _DONOR_LANGUAGE_TO_TARGET_FAMILY_RULES:
        if re.search(pattern, lang):
            return family
    return None


def _resolve_international_or_wanderword_family(other_donor_languages: object) -> str:

    if other_donor_languages is None or pd.isna(other_donor_languages):
        return "Other"

    parts = [
        p.strip()
        for p in str(other_donor_languages).split(";")
        if p and str(p).strip()
    ]
    if not parts:
        return "Other"

    for lang in parts:
        mapped = _map_donor_language_to_target_family(lang)
        if mapped == "Latin":
            return "Latin"

    mapped_last = _map_donor_language_to_target_family(parts[-1])
    return mapped_last or "Other"


def build_binary_loan_dataset(df: pd.DataFrame,
    exclude_old_words: bool = False, min_first_year: int | None = None,
    exclude_international_wanderword: bool = False) -> pd.DataFrame:

    df2 = filter_real_entries(
        df,
        exclude_old_words=exclude_old_words,
        min_first_year=min_first_year,
    )
    df2 = add_custom_labels(df2)

    if exclude_international_wanderword:
        df2 = df2[~df2["origin_clean"].isin(["international", "wanderword"])].copy()

    print("\nFormation_type distribution after filter:")
    if "formation_type" in df2.columns:
        print(df2["formation_type"].value_counts(dropna=False).to_string())

    print("\nOrigin_clean x is_loan_custom:")
    print(df2.groupby("origin_clean")["is_loan_custom"]
        .value_counts(dropna=False).sort_index())

    df_bin = df2[df2["is_loan_custom"].notna()].copy()
    df_bin["y_loan"] = df_bin["is_loan_custom"].astype(int)

    print("\nBinary dataset size:", len(df_bin))
    print(df_bin["y_loan"].value_counts())

    return df_bin
