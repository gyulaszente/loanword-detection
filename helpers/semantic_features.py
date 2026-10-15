"""Semantic features for etymology classification.
Each lemma is assigned an OdeNet / Princeton WordNet lexicographer-file label
using its German translations
If no usable German lookup is found, the entry receives 'MISSING'.
OdeNet github repo: https://github.com/hdaSprachtechnologie/odenet
"""

from __future__ import annotations

import warnings
from collections import Counter
from functools import lru_cache
from typing import Any

import pandas as pd


# The 45 Princeton WordNet lexicographer files
LEXFILES: list[str] = [
    "noun.Tops", "noun.act", "noun.animal", "noun.artifact", "noun.attribute",
    "noun.body", "noun.cognition", "noun.communication", "noun.event",
    "noun.feeling", "noun.food", "noun.group", "noun.location", "noun.motive",
    "noun.object", "noun.person", "noun.phenomenon", "noun.plant",
    "noun.possession", "noun.process", "noun.quantity", "noun.relation",
    "noun.shape", "noun.state", "noun.substance", "noun.time",
    "verb.body", "verb.change", "verb.cognition", "verb.communication",
    "verb.competition", "verb.consumption", "verb.contact", "verb.creation",
    "verb.emotion", "verb.motion", "verb.perception", "verb.possession",
    "verb.social", "verb.stative", "verb.weather",
    "adj.all", "adj.pert", "adj.ppl", "adv.all",
]

SEMANTIC_FIELD_MISSING = "MISSING"
ODENET_SPEC = "odenet:1.3"
EN_WORDNET_SPEC = "omw-en31:1.4"



@lru_cache(maxsize=1)
def _get_wn_modules() -> tuple[Any, Any] | tuple[None, None]:
    """Return (de_wn, en_wn) or (None, None) if wn resources are unavailable."""
    try:
        import wn

        de = wn.Wordnet(ODENET_SPEC)
        en = wn.Wordnet(EN_WORDNET_SPEC)
        return de, en
    except Exception as e:
        warnings.warn(f"wn resources unavailable: {e}")
        return None, None


@lru_cache(maxsize=None)
def _odenet_lookup_lexfiles(word: str) -> tuple[str, ...]:
    """
    Return broad WordNet meaning labels for one German word.
    For each German gloss for the same word, duplicate labels
    are removed so one translation does not get extra weight
    just because it has many OdeNet senses.
    """
    de, en = _get_wn_modules()
    if de is None:
        return ()

    word_clean = word.strip()
    labels: list[str] = []
    seen: set[str] = set()
    seen_word_ids: set[str] = set()
    for candidate in [word_clean, word_clean.capitalize()]:
        words = de.words(candidate)
        if not words:
            continue
        for w in words:
            word_id = getattr(w, "id", None) or repr(w)
            if word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            senses = list(w.senses())
            if not senses:
                continue
            for sense in senses:
                syn = sense.synset()
                ili = syn.ili
                if not ili:
                    continue
                en_syns = en.synsets(ili=ili)
                if not en_syns:
                    continue
                lexfile = en_syns[0].lexfile()
                if lexfile and lexfile not in seen:
                    labels.append(lexfile)
                    seen.add(lexfile)
    return tuple(labels)


def _majority_label(labels: list[str]) -> str | None:
    """Return the most frequent label. If more than one label has the most
    occurances, choose the one that appeared first."""
    if not labels:
        return None
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], labels.index(label)))


def lookup_semantic_field(german_translation: str | None) -> str:
    """
    Return an OdeNet / WordNet lexfile label for one entry.
    The function tries all one-word German translations via OdeNet. Each matched
    translation contributes all usable Princeton WordNet lexfile labels from its
    linked OdeNet senses, with repeated labels from the same translation counted
    only once. The final label is chosen by majority vote across translations;
    ties are resolved by the order of translations and senses. If no usable
    match is found, 'MISSING' is returned.
    """
    if german_translation:
        parts = [p.strip() for p in str(german_translation).split("|") if p.strip()]
        labels: list[str] = []
        for part in parts:
            if " " in part:
                continue
            labels.extend(_odenet_lookup_lexfiles(part))

        label = _majority_label(labels)
        if label:
            return label

    return SEMANTIC_FIELD_MISSING


def assign_semantic_fields(
    df: pd.DataFrame, *,
    translation_col: str = "translation",
    output_col: str = "semantic_field",) -> pd.DataFrame:
    """Add an OdeNet-derived semantic-field column to *df* and return a copy."""
    out = df.copy()
    translations = (
        out[translation_col].fillna("").astype(str)
        if translation_col in out.columns
        else pd.Series("", index=out.index))

    fields: list[str] = []
    for tr in translations:
        fields.append(lookup_semantic_field(tr))
    out[output_col] = fields
    return out


def semantic_field_features(
    df: pd.DataFrame, *,
    semantic_field_col: str = "semantic_field",
    fields: list[str] | None = None,
    prefix: str = "sem_") -> pd.DataFrame:
    """One-hot encoded DataFrame of OdeNet / WordNet lexfile features."""
    if fields is None:
        fields = LEXFILES

    out = pd.DataFrame(index=df.index)
    col = (df[semantic_field_col]
        if semantic_field_col in df.columns
        else pd.Series(SEMANTIC_FIELD_MISSING, index=df.index))

    for field in fields:
        col_name = (prefix
            + field.lower().replace(" ", "_").replace("/", "_").replace(".", "_"))
        out[col_name] = (col == field).astype(int)

    out[prefix + "missing"] = (col == SEMANTIC_FIELD_MISSING).astype(int)
    return out


def semantic_field_coverage_report(
    df: pd.DataFrame, *,
    semantic_field_col: str = "semantic_field",
    group_col: str | None = None) -> pd.DataFrame:
    """
    Count how many rows received each semantic field label.
    By default, the function returns one row per semantic field with a raw count
    and percentage of the dataframe. This is useful for checking how much
    semantic information was assigned and how much is still missing.
    If group_col is given, the function instead returns a cross-tabulation. For
    example, group_col="origin_type" shows how often each semantic field appears
    within each origin type.
    """
    work = df.copy()
    if semantic_field_col not in work.columns:
        raise ValueError(
            f"Column {semantic_field_col!r} not found. Run assign_semantic_fields() first.")

    if group_col and group_col in work.columns:
        table = pd.crosstab(work[semantic_field_col], work[group_col])
        table["Total"] = table.sum(axis=1)
        return table.sort_values("Total", ascending=False)

    counts = work[semantic_field_col].value_counts(dropna=False).rename("count")
    result = counts.reset_index()
    result.columns = [semantic_field_col, "count"]
    result["pct"] = (result["count"] / len(work) * 100).round(1)
    return result
