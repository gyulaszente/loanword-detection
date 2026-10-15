"""
Logistic-regression classification helpers for the ÚESz loanword experiments.

"""



from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    classification_report,
    log_loss,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split, StratifiedKFold

from .phonology import (
    V_PREFIXES,
    add_custom_labels,
    build_binary_loan_dataset,
    filter_real_entries,
    make_ngram_documents,
)

from .ipa_enrichment import phonemize_lemmas
from .emmorph_pos import SEGMENT_RE, emmorph_analyse_many
from .semantic_features import assign_semantic_fields, semantic_field_features

# Constants

""" NOT USED IN THE STATISTICS """
CERTAINTY_TO_WEIGHT: dict[float, float] = {
    0.0: 1.0,   # attested
    1.0: 0.8,   # more probable
    2.0: 0.4,   # less probable
    3.0: 0.2,   # uncertain
}



def _safe_metric_label(label: str) -> str:
    """Normalize a class label for use in summary-metric keys."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_").lower()



def _resolve_semantic_feature_mode(*, semantic_feature: str | None = None,
    add_semantic_features: bool | None = None) -> str | None:

    if semantic_feature is not None:
        mode = str(semantic_feature).strip().lower()
        if mode in {"", "none"}:
            return None
        if mode == "odenet":
            return "odenet"
        raise ValueError(
            "semantic_feature must be one of None or 'odenet'. "
            f"Got {semantic_feature!r}."
        )

    if add_semantic_features:
        return "odenet"
    return None


POS_FEATURE_ORDER: tuple[str, ...] = (
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "INTJ",
    "DET",
    "NUM",
    "ADP",
    "CCONJ",
    "PART",
    "OTHER",
    "MISSING",
)

WORD_SCALAR_NAMES: tuple[str, ...] = (
    "word_length",
    "vowel_ratio",
    "long_ratio",
    "voiced_cons_ratio",
)

WORD_SCALAR_COUNT = len(WORD_SCALAR_NAMES)

_PANPHON_FEATURE_TABLE = None
_PANPHON_FEATURE_NAMES: list[str] | None = None


def _primary_pos_label(pos_value: object) -> str:
    """Return the first POS label from emMorph-style POS cells."""
    if pos_value is None or (isinstance(pos_value, float) and pd.isna(pos_value)):
        return ""
    text = str(pos_value).strip()
    if not text:
        return ""
    return text.replace("|", ";").split(";")[0].strip().upper()


def _validate_articulatory_backend(feature_backend: str) -> str:
    """Normalise and validate the articulatory feature backend name."""
    backend = str(feature_backend).strip().lower()
    if backend != "panphon":
        raise ValueError("articulatory_backend must be None or 'panphon'")
    return backend

def _patch_panphon_utf8_reader() -> None:
    """
    Patch PanPhon so its bundled CSV tables are read as UTF-8 on Windows.
    Without this, PanPhon can try to read its IPA tables with the active
    Windows code page and fail before the classifier starts.
    """
    from importlib.resources import files
    import panphon.featuretable as panphon_featuretable

    if getattr(panphon_featuretable.FeatureTable._read_bases, "_utf8_reader_patch", False):
        return

    def _read_bases_utf8(self, fn: str, weights):
        spec_to_int = {"+": 1, "0": 0, "-": -1}
        with files("panphon").joinpath(fn).open(encoding="utf-8") as f:
            df = pd.read_csv(f)

        df["ipa"] = df["ipa"].apply(self.normalize)
        feature_names = list(df.columns[1:])
        df[feature_names] = df[feature_names].map(lambda x: spec_to_int[x])
        segments = [
            (row["ipa"], panphon_featuretable.Segment(feature_names, row[1:].to_dict(), weights=weights))
            for (_, row) in df.iterrows()
        ]
        seg_dict = dict(segments)
        return segments, seg_dict, feature_names

    _read_bases_utf8._utf8_reader_patch = True
    panphon_featuretable.FeatureTable._read_bases = _read_bases_utf8



def _get_panphon_feature_table():
    """Lazily construct a patched PanPhon feature table."""
    global _PANPHON_FEATURE_TABLE, _PANPHON_FEATURE_NAMES
    if _PANPHON_FEATURE_TABLE is None:
        _patch_panphon_utf8_reader()
        import panphon

        _PANPHON_FEATURE_TABLE = panphon.FeatureTable()
        _PANPHON_FEATURE_NAMES = list(_PANPHON_FEATURE_TABLE.names)
    return _PANPHON_FEATURE_TABLE


def get_articulatory_feature_names(feature_backend: str = "panphon") -> list[str]:
    """Return the PanPhon articulatory feature names."""
    _validate_articulatory_backend(feature_backend)
    _get_panphon_feature_table()
    return list(_PANPHON_FEATURE_NAMES or [])


def get_articulatory_feature_dim(feature_backend: str = "panphon") -> int:
    """Return the number of PanPhon articulatory dimensions."""
    return len(get_articulatory_feature_names(feature_backend))


def get_total_feature_dim(feature_backend: str = "panphon") -> int:
    """Return the total dense feature dimension for PanPhon plus word scalars."""
    return get_articulatory_feature_dim(feature_backend) + WORD_SCALAR_COUNT


def _panphon_token_matrix(ipa: str) -> np.ndarray:
    """Convert an IPA string to PanPhon feature vectors, one row per segment."""
    ft = _get_panphon_feature_table()
    vectors = ft.word_to_vector_list(ipa, numeric=True)
    if not vectors:
        return np.zeros((0, get_articulatory_feature_dim("panphon")), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)

def _compute_word_scalars(mat: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Compute word-level summary values from a segment-feature matrix."""
    if mat.size == 0:
        return np.zeros(WORD_SCALAR_COUNT, dtype=np.float32)

    feature_index = {name: i for i, name in enumerate(feature_names)}
    n = mat.shape[0]

    def positive(*names: str) -> np.ndarray:
        for name in names:
            if name in feature_index:
                return mat[:, feature_index[name]] > 0
        return np.zeros(n, dtype=bool)

    vowel_mask = positive("vowel", "syl")
    long_mask = positive("long")
    voiced_mask = positive("voiced", "voi")

    n_vowels = int(vowel_mask.sum())
    n_long = int(long_mask.sum())
    n_cons = n - n_vowels
    n_v_cons = int((~vowel_mask & voiced_mask).sum())

    return np.array(
        [n / 15.0, n_vowels / n, n_long / n, n_v_cons / n_cons if n_cons > 0 else 0.0],
        dtype=np.float32,
    )


def _ipa_to_feature_matrix(
    ipa: str,
    feature_backend: str = "panphon") -> tuple[np.ndarray, list[str]]:
    """Convert an IPA string to a PanPhon per-segment feature matrix."""
    _validate_articulatory_backend(feature_backend)
    mat = _panphon_token_matrix(ipa)
    return mat, get_articulatory_feature_names("panphon")


def word_to_articulatory_vector(
    ipa: str,
    feature_backend: str = "panphon",
) -> np.ndarray:
    """Convert an IPA string into one dense word-level articulatory vector."""
    mat, feature_names = _ipa_to_feature_matrix(ipa, feature_backend=feature_backend)
    if mat.size == 0:
        return np.zeros(get_total_feature_dim(feature_backend), dtype=np.float32)

    mean_vec = mat.mean(axis=0)
    extra = _compute_word_scalars(mat, feature_names)
    return np.concatenate([mean_vec, extra])


def build_articulatory_matrix(
    df: pd.DataFrame,
    feature_backend: str = "panphon",
) -> np.ndarray:
    """Build a dense articulatory feature matrix from the 'ipa' column."""
    return np.array(
        [
            word_to_articulatory_vector(str(ipa), feature_backend=feature_backend)
            for ipa in df["ipa"]
        ],
        dtype=np.float32,
    )


# hidden helpers
def _filter_by_pos(
    df: pd.DataFrame,
    pos_filter: str | list[str] | None,
    pos_col: str = "pos",
) -> pd.DataFrame:
    """
    Restrict df to rows whose primary POS tag matches 'pos_filter'.
    Matching is case-insensitive and uses the first POS tag only. POS
    cells may contain semicolon-separated emMorph readings such as
    '"NOUN;VERB"' 
    --> THIS IS NOT USED IN THE THESIS, IT WAS ONLY USED FOR EXPERIMENTS
    """
    if pos_filter is None:
        return df
    if isinstance(pos_filter, str):
        allowed = {pos_filter.upper()}
    else:
        allowed = {p.upper() for p in pos_filter}
    if pos_col not in df.columns:
        raise ValueError(f"POS column '{pos_col}' not found in DataFrame")

    pos_main = df[pos_col].map(_primary_pos_label)
    mask = pos_main.isin(allowed)
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"pos_filter={pos_filter!r} matched no rows. "
            f"Available POS values: {sorted(pos_main.unique())}"
        )
    return filtered


def _build_pos_feature_matrix(
    df: pd.DataFrame,
    include_missing_pos: bool = True,
    pos_col: str = "pos",
) -> tuple[scipy.sparse.csr_matrix, list[str]]:
    """
    Build a multi-hot POS feature matrix aligned to 'df' rows.
    POS cells may contain semicolon-separated emMorph readings such as
    'NOUN;ADJ'. In that case, both 'pos_noun' and 'pos_adj' are activated.
    Older pipe-separated cells are also accepted. The UPOS-style tags emitted
    by 'emmorph_pos.py' are retained explicitly; unknown tags become
    'OTHER'. Empty values become 'MISSING' when 'include_missing_pos' is
    true, otherwise they fall into 'OTHER'.
    """
    if pos_col not in df.columns:
        raise ValueError(f"POS column '{pos_col}' not found in DataFrame")

    keep = set(POS_FEATURE_ORDER) - {"OTHER", "MISSING"}
    feature_names = ["pos_" + p.lower() for p in POS_FEATURE_ORDER if include_missing_pos or p != "MISSING"]
    categories = [p for p in POS_FEATURE_ORDER if include_missing_pos or p != "MISSING"]
    category_to_code = {category: code for code, category in enumerate(categories)}

    row_indices: list[int] = []
    col_indices: list[int] = []
    for row_idx, pos_value in enumerate(df[pos_col]):
        labels = [
            label.strip().upper()
            for label in str(pos_value or "").replace("|", ";").split(";")
            if label.strip()
        ]
        if not labels:
            labels = ["MISSING" if include_missing_pos else "OTHER"]

        mapped_labels: list[str] = []
        for label in labels:
            if label in keep:
                mapped_labels.append(label)
            elif label == "MISSING" and include_missing_pos:
                mapped_labels.append("MISSING")
            else:
                mapped_labels.append("OTHER")

        for label in dict.fromkeys(mapped_labels):
            code = category_to_code.get(label)
            if code is not None:
                row_indices.append(row_idx)
                col_indices.append(code)

    mat = scipy.sparse.csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.float32),
            (np.array(row_indices, dtype=np.int32), np.array(col_indices, dtype=np.int32)),
        ),
        shape=(len(df), len(categories)),
        dtype=np.float32,
    )
    return mat, feature_names

def extract_vowel_coefs(
    vec: CountVectorizer,
    clf: LogisticRegression,
    v_prefixes: tuple[str, ...] = V_PREFIXES,
) -> dict[str, float]:

    feat = vec.get_feature_names_out()
    coef = clf.coef_[0]  # works for binary

    return {
        f: float(w)
        for f, w in zip(feat, coef)
        if f.startswith(v_prefixes)
    }


def summarize_coef_maps(
    coef_maps: list[dict[str, float]]) -> dict[str, dict[str, float]]:

    all_feats = sorted({f for m in coef_maps for f in m})

    summary: dict[str, dict[str, float]] = {}
    for f in all_feats:
        vals = np.array([m.get(f, np.nan) for m in coef_maps], dtype=float)
        present = vals[~np.isnan(vals)]
        if len(present) == 0:
            continue
        summary[f] = {
            "mean": float(np.mean(present)),
            "std": float(np.std(present, ddof=1)) if len(present) > 1 else 0.0,
            "min": float(np.min(present)),
            "max": float(np.max(present)),
            "n_present": int(len(present)),
        }
    return summary


def _safe_rate(mask: pd.Series, condition: pd.Series) -> float:
    """Return the mean of *condition* over *mask*, or NaN if the mask is empty."""
    denom = int(mask.sum())
    if denom == 0:
        return float("nan")
    return float(condition[mask].mean())


def _binary_group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:

    rows: list[dict[str, float | int | str]] = []
    for group_value, sub in df.groupby(group_col, dropna=False):
        true_loan = sub["y_true"] == 1
        true_native = sub["y_true"] == 0
        pred_loan = sub["y_pred"] == 1
        pred_native = sub["y_pred"] == 0

        rows.append(
            {
                group_col: group_value,
                "n": int(len(sub)),
                "n_true_loan": int(true_loan.sum()),
                "n_true_native": int(true_native.sum()),
                "accuracy": float(sub["correct"].mean()),
                "error_rate": float((~sub["correct"]).mean()),
                "predicted_loan_rate": float(pred_loan.mean()),
                "predicted_native_rate": float(pred_native.mean()),
                "loan_as_native_rate": _safe_rate(true_loan, pred_native),
                "native_as_loan_rate": _safe_rate(true_native, pred_loan),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["accuracy", "n"], ascending=[False, False]).reset_index(drop=True)
    return out

def score_binary_classifier_table(
    df_rows: pd.DataFrame,
    df_source: pd.DataFrame,
    clf: LogisticRegression,
    vec: CountVectorizer,
    *,
    ns: tuple[int, ...] = (2, 3),
    add_vowel_string: bool = True,
    add_vowel_patterns: bool = True,
    add_vowel_length_string: bool = True,
    articulatory_backend: str | None = None,
    semantic_feature: str | None = None,
    add_semantic_features: bool = False,
    add_pos_features: bool = True,
    include_missing_pos: bool = True,
    strip_final_ik: bool = False,) -> pd.DataFrame:

    merge_cols = [
        col
        for col in ("lemma", "ipa", "origin_type", "primary_donor_family")
        if col in df_rows.columns and col in df_source.columns]
    if not merge_cols:
        raise ValueError("No shared merge columns found between df_rows and df_source")

    needed_source_cols = [col
        for col in (
            "lemma",
            "ipa",
            "origin_type",
            "primary_donor_family",
            "pos",
            "first_year",)
        if col in df_source.columns]

    source_meta = df_source[needed_source_cols].drop_duplicates().copy()
    scored = df_rows.merge(source_meta, on=merge_cols, how="left", suffixes=("", "_src")).copy()

    docs = make_ngram_documents(
        scored,
        ns=ns,
        add_vowel_string=add_vowel_string,
        add_vowel_patterns=add_vowel_patterns,
        add_vowel_length_string=add_vowel_length_string,
        strip_final_ik=strip_final_ik)

    X_ng = vec.transform(docs).astype(np.float32)
    X = X_ng
    expected_total = getattr(clf, "n_features_in_", X_ng.shape[1])
    expected_extra = expected_total - X_ng.shape[1]
    used_extra = 0

    if articulatory_backend is not None:
        articulatory_backend = _validate_articulatory_backend(articulatory_backend)
        X_art_full = scipy.sparse.csr_matrix(
            build_articulatory_matrix(
                scored,
                feature_backend=articulatory_backend,
            ).astype(np.float32)
        )
        art_dim = min(X_art_full.shape[1], max(expected_extra - used_extra, 0))
        if art_dim > 0:
            X = scipy.sparse.hstack([X, X_art_full[:, :art_dim]], format="csr")
            used_extra += art_dim

    semantic_mode = _resolve_semantic_feature_mode(
        semantic_feature=semantic_feature,
        add_semantic_features=add_semantic_features)

    if semantic_mode is not None:
        scored_sem = assign_semantic_fields(scored)
        X_sem_full = scipy.sparse.csr_matrix(
            semantic_field_features(scored_sem).to_numpy(dtype=np.float32))
        sem_dim = min(X_sem_full.shape[1], max(expected_extra - used_extra, 0))
        if sem_dim > 0:
            X = scipy.sparse.hstack([X, X_sem_full[:, :sem_dim]], format="csr")
            used_extra += sem_dim

    if add_pos_features:
        X_pos_full, _ = _build_pos_feature_matrix(
            scored,
            include_missing_pos=include_missing_pos)
        pos_dim = min(X_pos_full.shape[1], max(expected_extra - used_extra, 0))
        if pos_dim > 0:
            X = scipy.sparse.hstack([X, X_pos_full[:, :pos_dim]], format="csr")
            used_extra += pos_dim

    if X.shape[1] != expected_total:
        raise ValueError(
            "Feature mismatch while scoring rows: "
            f"built {X.shape[1]} features, but the classifier expects {expected_total}. "
            "This usually means the fitted model was trained with different "
            "articulatory/semantic/POS settings than the scoring call.")

    prob_loan = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)
    decision = clf.decision_function(X)

    scored["prob_loan"] = prob_loan
    scored["prob_native"] = 1.0 - prob_loan
    scored["decision_score"] = decision
    scored["y_pred_from_model"] = y_pred
    scored["pred_label"] = np.where(scored["y_pred_from_model"] == 1, "loan", "native")

    if "y_true" in scored.columns:
        scored["true_label"] = np.where(scored["y_true"] == 1, "loan", "native")
    if "y_pred" in scored.columns:
        scored["prediction_matches_eval"] = scored["y_pred"] == scored["y_pred_from_model"]

    return scored


def _score_binary_rows_direct(
    df_rows: pd.DataFrame,
    clf: LogisticRegression,
    vec: CountVectorizer,
    *,
    ns: tuple[int, ...] = (2, 3),
    add_vowel_string: bool = True,
    add_vowel_patterns: bool = True,
    add_vowel_length_string: bool = True,
    articulatory_backend: str | None = None,
    semantic_feature: str | None = None,
    add_semantic_features: bool = False,
    add_pos_features: bool = True,
    include_missing_pos: bool = True,
    strip_final_ik: bool = False,) -> pd.DataFrame:
    """Score already-prepared rows without merging them against a source table."""
    scored = df_rows.copy()
    docs = make_ngram_documents(
        scored,
        ns=ns,
        add_vowel_string=add_vowel_string,
        add_vowel_patterns=add_vowel_patterns,
        add_vowel_length_string=add_vowel_length_string,
        strip_final_ik=strip_final_ik)

    X_ng = vec.transform(docs).astype(np.float32)
    X = X_ng
    expected_total = getattr(clf, "n_features_in_", X_ng.shape[1])
    expected_extra = expected_total - X_ng.shape[1]
    used_extra = 0

    if articulatory_backend is not None:
        articulatory_backend = _validate_articulatory_backend(articulatory_backend)
        X_art_full = scipy.sparse.csr_matrix(
            build_articulatory_matrix(
                scored,
                feature_backend=articulatory_backend,
            ).astype(np.float32))
        art_dim = min(X_art_full.shape[1], max(expected_extra - used_extra, 0))
        if art_dim > 0:
            X = scipy.sparse.hstack([X, X_art_full[:, :art_dim]], format="csr")
            used_extra += art_dim

    semantic_mode = _resolve_semantic_feature_mode(
        semantic_feature=semantic_feature,
        add_semantic_features=add_semantic_features)
    if semantic_mode is not None:
        scored_sem = assign_semantic_fields(scored)
        X_sem_full = scipy.sparse.csr_matrix(
            semantic_field_features(scored_sem).to_numpy(dtype=np.float32))
        sem_dim = min(X_sem_full.shape[1], max(expected_extra - used_extra, 0))
        if sem_dim > 0:
            X = scipy.sparse.hstack([X, X_sem_full[:, :sem_dim]], format="csr")
            used_extra += sem_dim

    if add_pos_features:
        X_pos_full, _ = _build_pos_feature_matrix(scored,
            include_missing_pos=include_missing_pos)
        pos_dim = min(X_pos_full.shape[1], max(expected_extra - used_extra, 0))
        if pos_dim > 0:
            X = scipy.sparse.hstack([X, X_pos_full[:, :pos_dim]], format="csr")
            used_extra += pos_dim

    if X.shape[1] != expected_total:
        raise ValueError(
            "Feature mismatch while scoring rows: "
            f"built {X.shape[1]} features, but the classifier expects {expected_total}.")

    prob_loan = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)
    decision = clf.decision_function(X)

    scored["prob_loan"] = prob_loan
    scored["prob_native"] = 1.0 - prob_loan
    scored["decision_score"] = decision
    scored["y_pred_from_model"] = y_pred
    scored["pred_label"] = np.where(scored["y_pred_from_model"] == 1, "loan", "native")
    return scored


def _is_lexical_emmorph_tag(tag: str) -> bool:

    return str(tag or "").startswith("/")


def _emmorph_segments_with_affixes(
    analysis: str,) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split one emMorph analysis into lexical vs. non-lexical segments."""
    lexical: list[tuple[str, str]] = []
    affixes: list[tuple[str, str]] = []
    for surface, tag in SEGMENT_RE.findall(analysis or ""):
        if _is_lexical_emmorph_tag(tag):
            lexical.append((surface, tag))
        else:
            affixes.append((surface, tag))
    return lexical, affixes


def _normalize_surface(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def _pick_structured_emmorph_analysis(
    lemma: str,
    analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Prefer compound analyses, then derivational stem analyses, then whole-word.
    """
    if not analyses:
        return {
            "analysis": "",
            "structure_type": "whole_word",
            "based_on_parts": [lemma],}

    compound_candidate: dict[str, Any] | None = None
    deriv_candidate: dict[str, Any] | None = None
    fallback = str(analyses[0].get("analysis") or "")

    for item in analyses:
        analysis = str(item.get("analysis") or "")
        lexical, affixes = _emmorph_segments_with_affixes(analysis)
        lexical_surfaces = [surface for surface, _tag in lexical if surface]

        if len(lexical_surfaces) >= 2 and compound_candidate is None:
            compound_candidate = {
                "analysis": analysis,
                "structure_type": "compound",
                "based_on_parts": lexical_surfaces,
            }
            continue

        if len(lexical_surfaces) == 1 and affixes and deriv_candidate is None:
            base = lexical_surfaces[0]
            if _normalize_surface(base) != _normalize_surface(lemma):
                deriv_candidate = {
                    "analysis": analysis,
                    "structure_type": "derivation",
                    "based_on_parts": [base]}

    if compound_candidate is not None:
        return compound_candidate
    if deriv_candidate is not None:
        return deriv_candidate
    return {
        "analysis": fallback,
        "structure_type": "whole_word",
        "based_on_parts": [lemma]}


def _build_gold_part_label_map(df_source: pd.DataFrame) -> dict[str, str | None]:

    labeled = add_custom_labels(df_source)
    if "lemma" not in labeled.columns:
        return {}

    gold_map: dict[str, str | None] = {}
    lemma_series = labeled["lemma"].astype(str).str.strip().str.lower()
    for lemma, group in labeled.groupby(lemma_series, sort=False):
        usable = group["is_loan_custom"].dropna().unique().tolist()
        if len(usable) == 1:
            gold_map[str(lemma)] = "loan" if bool(usable[0]) else "native"
        else:
            gold_map[str(lemma)] = None
    return gold_map


def balance_indices_equalize(
    y: np.ndarray,
    idx: np.ndarray,
    rng: np.random.RandomState,
    method: str = "undersample") -> np.ndarray:

    y_sub = y[idx]
    classes, counts = np.unique(y_sub, return_counts=True)

    if method == "undersample":
        target_n = counts.min()
        replace = False
    elif method == "oversample":
        target_n = counts.max()
        replace = True
    else:
        raise ValueError("method must be 'undersample' or 'oversample'")

    picked: list[np.ndarray] = []
    for c in classes:
        c_idx = idx[y_sub == c]
        picked.append(rng.choice(c_idx, size=target_n, replace=replace))

    balanced = np.concatenate(picked)
    rng.shuffle(balanced)
    return balanced





# Binary classifier  –  loan vs. native

def train_test_loan_classifier(
    df: pd.DataFrame,
    ns: tuple[int, ...] = (2, 3, 4, 5),
    exclude_old_words: bool = False,
    exclude_international_wanderword: bool = False,
    add_vowel_string: bool = False,
    add_vowel_patterns: bool = True,
    add_vowel_length_string: bool = True,
    articulatory_backend: str | None = None,
    semantic_feature: str | None = None,
    add_semantic_features: bool = False,
    add_pos_features: bool = False,
    include_missing_pos: bool = True,
    skip_initial_unbounded: bool = False,
    strip_final_ik: bool = True,
    min_df_grid: int | tuple[int, ...] = 2,
    random_state: int = 123,
    test_size: float = 0.2,
    mode: str = "single",       # "single" | "repeated_holdout" | "kfold"
    n_runs: int = 5,            # for repeated_holdout
    k: int = 5,                 # for kfold
    certainty_to_weight: dict | None = None,
    min_first_year: int | None = None,
    pos_filter: str | list[str] | None = None,
    equalize_classes: bool = False,
    equalize_method: str = "undersample",   # "undersample" | "oversample"
    verbose: bool = True):

    if isinstance(min_df_grid, int):
        min_df_grid = (min_df_grid,)
    else:
        min_df_grid = tuple(min_df_grid)

    all_vowel_maps: list[dict] = []

    df_bin = build_binary_loan_dataset(
        df,
        exclude_old_words=exclude_old_words,
        min_first_year=min_first_year,
        exclude_international_wanderword=exclude_international_wanderword)
    df_bin = _filter_by_pos(df_bin, pos_filter)

    docs = make_ngram_documents(
        df_bin,
        ns=ns,
        add_vowel_string=add_vowel_string,
        add_vowel_patterns=add_vowel_patterns,
        add_vowel_length_string=add_vowel_length_string,
        skip_initial_unbounded=skip_initial_unbounded,
        strip_final_ik=strip_final_ik)
    X_art_all = None
    if articulatory_backend is not None:
        articulatory_backend = _validate_articulatory_backend(articulatory_backend)
        X_art_all = build_articulatory_matrix(
            df_bin,
            feature_backend=articulatory_backend,
        )
    X_sem_all = None
    semantic_feature_names: list[str] = []
    semantic_mode = _resolve_semantic_feature_mode(
        semantic_feature=semantic_feature,
        add_semantic_features=add_semantic_features,
    )
    if semantic_mode is not None:
        df_bin = assign_semantic_fields(df_bin)
        X_sem_df = semantic_field_features(df_bin)
        semantic_feature_names = list(X_sem_df.columns)
        X_sem_all = scipy.sparse.csr_matrix(X_sem_df.to_numpy(dtype=np.float32))
    X_pos_all = None
    pos_feature_names: list[str] = []
    if add_pos_features:
        X_pos_all, pos_feature_names = _build_pos_feature_matrix(
            df_bin,
            include_missing_pos=include_missing_pos,
        )

    if verbose:
        sample_doc = docs.iloc[0]
        vowel_toks = [t for t in sample_doc.split() if t.startswith("V")]
        expects_vowels = add_vowel_string or add_vowel_patterns or add_vowel_length_string

        print("\nExample feature document:")
        print(sample_doc[:300])
        print("\nVowel tokens:", vowel_toks[:20])

        if expects_vowels and not vowel_toks:
            raise RuntimeError("Vowel features were requested, but no vowel tokens were generated.")

        if X_art_all is not None:
            print(f"Added PanPhon features: {X_art_all.shape[1]} columns")
        if X_sem_all is not None:
            print(f"Added semantic features: {X_sem_all.shape[1]} columns")
        if X_pos_all is not None:
            print(f"Added POS features: {X_pos_all.shape[1]} columns")


    y = df_bin["y_loan"].astype(int).values
    idx_all = df_bin.index.to_numpy()

    # fit and evaluate one train/test split
    def fit_one_split(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        split_label: str = "split",
    ) -> tuple[dict, pd.DataFrame]:

        rng = np.random.RandomState(random_state + hash(split_label) % 10_000)

        if verbose:
            cls, cnt = np.unique(y[train_idx], return_counts=True)
            print(f"\n[{split_label}] Train dist BEFORE:", dict(zip(cls, cnt)))

        if equalize_classes:
            train_idx = balance_indices_equalize(
                y=y, idx=train_idx, rng=rng, method=equalize_method
            )
            if verbose:
                cls, cnt = np.unique(y[train_idx], return_counts=True)
                print(f"[{split_label}] Train dist AFTER:", dict(zip(cls, cnt)))

        docs_train = docs.iloc[train_idx]
        docs_test = docs.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if verbose:
            print(f"\n--- {split_label.upper()} ---")
            print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
            uniq, cnt = np.unique(y_train, return_counts=True)
            print("Train dist:", dict(zip(uniq, cnt)))
            uniq, cnt = np.unique(y_test, return_counts=True)
            print("Test dist:", dict(zip(uniq, cnt)))

        best_local: dict = {
            "min_df": None, "macro_f1": -1.0, "log_loss": np.inf,
            "clf": None, "vec": None, "y_pred": None, "y_prob": None,
            "decision": None, "n_features": None}

        """ build the actual numeric feature matrices """
        for md in min_df_grid:
            vec = CountVectorizer(token_pattern=r"[^ ]+", binary=True, min_df=md)
            X_train_ng = vec.fit_transform(docs_train).astype(np.float32)
            X_test_ng = vec.transform(docs_test).astype(np.float32)
            X_train = X_train_ng
            X_test = X_test_ng
            
            if X_art_all is not None:
                X_train_art = scipy.sparse.csr_matrix(X_art_all[train_idx].astype(np.float32))
                X_test_art = scipy.sparse.csr_matrix(X_art_all[test_idx].astype(np.float32))
                X_train = scipy.sparse.hstack([X_train_ng, X_train_art], format="csr")
                X_test = scipy.sparse.hstack([X_test_ng, X_test_art], format="csr")
            if X_sem_all is not None:
                X_train_sem = X_sem_all[train_idx]
                X_test_sem = X_sem_all[test_idx]
                X_train = scipy.sparse.hstack([X_train, X_train_sem], format="csr")
                X_test = scipy.sparse.hstack([X_test, X_test_sem], format="csr")
            if X_pos_all is not None:
                X_train_pos = X_pos_all[train_idx]
                X_test_pos = X_pos_all[test_idx]
                X_train = scipy.sparse.hstack([X_train, X_train_pos], format="csr")
                X_test = scipy.sparse.hstack([X_test, X_test_pos], format="csr")

            clf = LogisticRegression(max_iter=4000, class_weight="balanced")

            if certainty_to_weight is not None and "certainty_level" in df_bin.columns:
                train_df_idx = idx_all[train_idx]
                sw = (
                    df_bin.loc[train_df_idx, "certainty_level"]
                    .map(certainty_to_weight)
                    .fillna(1.0)
                    .values
                )

                clf.fit(X_train, y_train, sample_weight=sw)
            else:

                clf.fit(X_train, y_train)


            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            decision = clf.decision_function(X_test)
            macro = f1_score(y_test, y_pred, average="macro")
            loss = log_loss(y_test, y_prob, labels=[0, 1])

            if verbose:
                print(
                    f"min_df={md}  features={X_train.shape[1]}  "
                    f"macro_f1={macro:.4f}  log_loss={loss:.4f}"
                )
                print(classification_report(y_test, y_pred, digits=4))

            if macro > best_local["macro_f1"]:
                best_local.update(
                    min_df=md, macro_f1=macro, clf=clf, vec=vec,
                    log_loss=loss, y_pred=y_pred, y_prob=y_prob,
                    decision=decision, n_features=X_train.shape[1],
                )

        # Evaluation frame
        test_df_idx = idx_all[test_idx]
        df_eval = df_bin.loc[
            test_df_idx, ["lemma", "ipa", "origin_type", "primary_donor_family"]
        ].copy()
        df_eval["y_true"] = y_test
        df_eval["y_pred"] = best_local["y_pred"]
        df_eval["prob_native"] = best_local["y_prob"][:, 0]
        df_eval["prob_loan"] = best_local["y_prob"][:, 1]
        df_eval["decision_score"] = best_local["decision"]
        df_eval["correct"] = df_eval["y_true"] == df_eval["y_pred"]

        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_test, best_local["y_pred"], average="macro", zero_division=0
        )
        p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
            y_test, best_local["y_pred"], labels=[0, 1], average=None, zero_division=0
        )
        best_local["eval_metrics"] = {
            "accuracy": float(accuracy_score(y_test, best_local["y_pred"])),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f_macro),
            "log_loss": float(best_local["log_loss"]),
            "precision_native": float(p_cls[0]),
            "recall_native": float(r_cls[0]),
            "f1_native": float(f_cls[0]),
            "precision_loan": float(p_cls[1]),
            "recall_loan": float(r_cls[1]),
            "f1_loan": float(f_cls[1]),
        }

        if verbose:
            print(
                f"\nBest {split_label}: "
                f"min_df={best_local['min_df']}, "
                f"features={best_local['n_features']}, "
                f"macro_f1={best_local['macro_f1']:.4f}, "
                f"log_loss={best_local['log_loss']:.4f}"
            )

        best_local["vowel_coef_map"] = extract_vowel_coefs(
            best_local["vec"],
            best_local["clf"],
        )

        return best_local, df_eval


    # Mode dispatch
    all_scores: list[float] = []
    all_losses: list[float] = []
    all_metric_rows: list[dict] = []
    all_eval_tables: list[pd.DataFrame] = []
    best_overall: dict | None = None
    best_eval: pd.DataFrame | None = None

    if mode == "single":
        tr, te = train_test_split(
            np.arange(len(df_bin)), test_size=test_size,
            stratify=y, random_state=random_state,
        )
        best_overall, best_eval = fit_one_split(tr, te, "single")
        summary = {
            "mode": mode,
            "test_size": test_size,
            "min_df_grid": list(min_df_grid),
            "macro_f1": float(best_overall["macro_f1"]),
            "log_loss": float(best_overall["log_loss"]),
            "best_min_df": int(best_overall["min_df"]),
            **best_overall["eval_metrics"],
            "vowel_coef_summary": {k: {"mean": v, "std": 0.0, "min": v, "max": v, "n": 1}
                                   for k, v in best_overall["vowel_coef_map"].items()},
            "semantic_feature_names": semantic_feature_names,
            "pos_feature_names": pos_feature_names,
        }
        if verbose:
            print("\n=== SINGLE SUMMARY ===")
            print(
                f"  accuracy: {summary['accuracy']:.4f}\n"
                f"  macro_f1: {summary['macro_f1']:.4f}\n"
                f"  log_loss: {summary['log_loss']:.4f}\n"
                f"  best_min_df: {summary['best_min_df']}"
            )

    elif mode == "repeated_holdout":
        for r in range(n_runs):
            tr, te = train_test_split(
                np.arange(len(df_bin)), test_size=test_size,
                stratify=y, random_state=random_state + r,
            )
            bl, de = fit_one_split(tr, te, f"run {r + 1}/{n_runs}")
            all_vowel_maps.append(bl["vowel_coef_map"])
            all_scores.append(bl["macro_f1"])
            all_losses.append(bl["log_loss"])
            all_metric_rows.append(bl["eval_metrics"])
            all_eval_tables.append(de.copy())
            if best_overall is None or bl["macro_f1"] > best_overall["macro_f1"]:
                best_overall, best_eval = bl, de

        scores = np.array(all_scores)
        losses = np.array(all_losses, dtype=float)
        metrics_df = pd.DataFrame(all_metric_rows)
        summary = {
            "mode": mode, "test_size": test_size, "n_runs": n_runs,
            "min_df_grid": list(min_df_grid),
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "macro_f1_min": float(scores.min()),
            "macro_f1_max": float(scores.max()),
            "log_loss_mean": float(losses.mean()),
            "log_loss_std": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
            "log_loss_min": float(losses.min()),
            "log_loss_max": float(losses.max()),
            "accuracy_mean": float(metrics_df["accuracy"].mean()),
            "precision_macro_mean": float(metrics_df["precision_macro"].mean()),
            "recall_macro_mean": float(metrics_df["recall_macro"].mean()),
            "f1_macro_mean": float(metrics_df["f1_macro"].mean()),
            "precision_native_mean": float(metrics_df["precision_native"].mean()),
            "recall_native_mean": float(metrics_df["recall_native"].mean()),
            "f1_native_mean": float(metrics_df["f1_native"].mean()),
            "precision_loan_mean": float(metrics_df["precision_loan"].mean()),
            "recall_loan_mean": float(metrics_df["recall_loan"].mean()),
            "f1_loan_mean": float(metrics_df["f1_loan"].mean()),
            "fold_evals": all_eval_tables,
            "vowel_coef_summary": summarize_coef_maps(all_vowel_maps),
            "semantic_feature_names": semantic_feature_names,
            "pos_feature_names": pos_feature_names,
        }
        if verbose:
            print("\n=== REPEATED HOLDOUT SUMMARY ===")
            _print_summary(summary)

    elif mode == "kfold":
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        for fold, (tr, te) in enumerate(
            skf.split(np.zeros(len(df_bin)), y), start=1
        ):
            bl, de = fit_one_split(tr, te, f"fold {fold}/{k}")
            all_scores.append(bl["macro_f1"])
            all_losses.append(bl["log_loss"])
            all_metric_rows.append(bl["eval_metrics"])
            all_eval_tables.append(de.copy())
            all_vowel_maps.append(bl["vowel_coef_map"])
            if best_overall is None or bl["macro_f1"] > best_overall["macro_f1"]:
                best_overall, best_eval = bl, de

        scores = np.array(all_scores)
        losses = np.array(all_losses, dtype=float)
        metrics_df = pd.DataFrame(all_metric_rows)
        summary = {
            "mode": mode, "k": k,
            "min_df_grid": list(min_df_grid),
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "macro_f1_min": float(scores.min()),
            "macro_f1_max": float(scores.max()),
            "log_loss_mean": float(losses.mean()),
            "log_loss_std": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
            "log_loss_min": float(losses.min()),
            "log_loss_max": float(losses.max()),
            "accuracy_mean": float(metrics_df["accuracy"].mean()),
            "precision_macro_mean": float(metrics_df["precision_macro"].mean()),
            "recall_macro_mean": float(metrics_df["recall_macro"].mean()),
            "f1_macro_mean": float(metrics_df["f1_macro"].mean()),
            "precision_native_mean": float(metrics_df["precision_native"].mean()),
            "recall_native_mean": float(metrics_df["recall_native"].mean()),
            "f1_native_mean": float(metrics_df["f1_native"].mean()),
            "precision_loan_mean": float(metrics_df["precision_loan"].mean()),
            "recall_loan_mean": float(metrics_df["recall_loan"].mean()),
            "f1_loan_mean": float(metrics_df["f1_loan"].mean()),
            "fold_evals": all_eval_tables,
            "vowel_coef_summary": summarize_coef_maps(all_vowel_maps),
            "semantic_feature_names": semantic_feature_names,
            "pos_feature_names": pos_feature_names,
        }
        if verbose:
            print("\n=== K-FOLD SUMMARY ===")
            _print_summary(summary)

    else:
        raise ValueError("mode must be 'single', 'repeated_holdout', or 'kfold'")

    return (
        best_overall["clf"],
        best_overall["vec"],
        best_eval,
        best_overall["min_df"],
        summary,
    )







# -----------------------------------------------------------------------------------------


# Multiclass classifier  –  donor-family

def build_origin_source_dataset_multiclass(
    df: pd.DataFrame,
    target_families: list[str],
    exclude_old_words: bool = False,
    require_ipa: bool = True,
    min_first_year: int | None = None) -> pd.DataFrame:
    """
    Build the nine-class origin/source dataset used in the multinomial
    classification task.
    """
    df2 = filter_real_entries(
        df,
        exclude_old_words=exclude_old_words,
        min_first_year=min_first_year,
    )

    df2 = add_custom_labels(df2)
    df2 = df2[df2["is_loan_custom"].notna()].copy()

    major_loan_classes = {
        "Latin",
        "German",
        "Romance",
        "English",
        "Slavic",
        "Turkic",
    }

    def assign_origin_source(row):
        origin = str(row.get("origin_clean", "") or "").strip().lower()
        family = str(row.get("family_clean", "") or "").strip()

        if row["is_loan_custom"] == False:
            return "non-loan"

        origin_parts = {
            p.strip()
            for p in origin.split("|")
            if p.strip()
        }

        if origin_parts & {"international", "wanderword"}:
            return "international/wanderword"

        if family in major_loan_classes:
            return family

        return "Other loan"

    df2["family_clean"] = df2.apply(assign_origin_source, axis=1)
    df2 = df2[df2["family_clean"].isin(target_families)].copy()

    if require_ipa and "ipa" in df2.columns:
        df2 = df2[df2["ipa"].notna()].copy()
        df2 = df2[df2["ipa"].astype(str).str.strip() != ""].copy()

    print("Origin/source multinomial dataset size:", len(df2))
    print(df2["family_clean"].value_counts())

    return df2

def majority_baseline_from_fold_evals(
    fold_evals: list[pd.DataFrame],
    *,
    true_col: str = "true_family",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute a majority-class baseline from k-fold held-out prediction tables.
    """
    all_eval = pd.concat(
        [
            ev.assign(_fold=i)
            for i, ev in enumerate(fold_evals, start=1)
        ],
        ignore_index=False,
    )

    labels = sorted(all_eval[true_col].dropna().astype(str).unique())
    rows = []

    for fold_id in sorted(all_eval["_fold"].unique()):
        test = all_eval[all_eval["_fold"] == fold_id]
        train = all_eval[all_eval["_fold"] != fold_id]

        majority_class = train[true_col].astype(str).value_counts().idxmax()

        y_true = test[true_col].astype(str).to_numpy()
        y_pred = np.array([majority_class] * len(test))

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        rows.append({
            "fold": fold_id,
            "majority_class": majority_class,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
        })

    baseline = pd.DataFrame(rows)

    summary = pd.Series({
        "accuracy": baseline["accuracy"].mean(),
        "precision_macro": baseline["precision_macro"].mean(),
        "recall_macro": baseline["recall_macro"].mean(),
        "f1_macro": baseline["f1_macro"].mean(),
        "macro_f1_std": baseline["f1_macro"].std(ddof=1),
    })

    return baseline, summary



def train_test_family_classifier(
    df: pd.DataFrame,
    target_families: list[str],
    ns: tuple[int, ...] = (2, 3, 4, 5),
    exclude_old_words: bool = False,
    add_vowel_string: bool = True,
    add_vowel_patterns: bool = True,
    add_vowel_length_string: bool = True,
    articulatory_backend: str | None = None,
    semantic_feature: str | None = None,
    add_semantic_features: bool = False,
    add_pos_features: bool = False,
    include_missing_pos: bool = True,
    skip_initial_unbounded: bool = False,
    strip_final_ik: bool = True,
    min_df_grid: int | tuple[int, ...] = 3,
    random_state: int = 123,
    test_size: float = 0.2,
    mode: str = "kfold",        # "single" | "repeated_holdout" | "kfold"
    n_runs: int = 10,           # for repeated_holdout
    k: int = 5,                 # for kfold
    certainty_to_weight: dict | None = None,
    require_ipa: bool = True,
    verbose: bool = True,
    equalize_classes: bool = True,
    equalize_method: str = "undersample",   # "undersample" | "oversample" --> not used in the experiments
    min_first_year: int | None = None,
    pos_filter: str | list[str] | None = None):

    if isinstance(min_df_grid, int):
        min_df_grid = (min_df_grid,)
    else:
        min_df_grid = tuple(min_df_grid)

    all_vowel_maps: list[dict] = []

    df_fam = build_origin_source_dataset_multiclass(
        df,
        target_families=target_families,
        require_ipa=require_ipa,
        exclude_old_words=exclude_old_words,
        min_first_year=min_first_year,
    )
    df_fam = _filter_by_pos(df_fam, pos_filter)

    docs = make_ngram_documents(
        df_fam,
        ns=ns,
        add_vowel_string=add_vowel_string,
        add_vowel_patterns=add_vowel_patterns,
        add_vowel_length_string=add_vowel_length_string,
        skip_initial_unbounded=skip_initial_unbounded,
        strip_final_ik=strip_final_ik,
    )
    X_art_all = None
    if articulatory_backend is not None:
        articulatory_backend = _validate_articulatory_backend(articulatory_backend)
        X_art_all = build_articulatory_matrix(
            df_fam,
            feature_backend=articulatory_backend,
        )
    X_sem_all = None
    semantic_feature_names: list[str] = []
    semantic_mode = _resolve_semantic_feature_mode(
        semantic_feature=semantic_feature,
        add_semantic_features=add_semantic_features,
    )
    if semantic_mode is not None:
        df_fam = assign_semantic_fields(df_fam)
        X_sem_df = semantic_field_features(df_fam)
        semantic_feature_names = list(X_sem_df.columns)
        X_sem_all = scipy.sparse.csr_matrix(X_sem_df.to_numpy(dtype=np.float32))
    X_pos_all = None
    pos_feature_names: list[str] = []
    if add_pos_features:
        X_pos_all, pos_feature_names = _build_pos_feature_matrix(
            df_fam,
            include_missing_pos=include_missing_pos,
        )

    if verbose:
        sample_doc = docs.iloc[0]
        vowel_toks = [t for t in sample_doc.split() if t.startswith("V")]
        expects_vowels = add_vowel_string or add_vowel_patterns or add_vowel_length_string

        print("\nExample feature document:")
        print(sample_doc[:300])
        print("\nVowel tokens:", vowel_toks[:30])

        if expects_vowels and not vowel_toks:
            raise RuntimeError("Vowel features were requested, but no vowel tokens were generated.")

        if X_art_all is not None:
            print(f"Added PanPhon features: {X_art_all.shape[1]} columns")
        if X_sem_all is not None:
            print(f"Added semantic features: {X_sem_all.shape[1]} columns")
        if X_pos_all is not None:
            print(f"Added POS features: {X_pos_all.shape[1]} columns")


    y = df_fam["family_clean"].astype(str).values
    class_labels = np.array(sorted(pd.unique(y)))
    idx_all = df_fam.index.to_numpy()

    # Fit one split
    def fit_one_split(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        split_label: str = "split",
    ) -> tuple[dict, pd.DataFrame]:

        # Optional class equalization (train set only)
        rng = np.random.RandomState(random_state + hash(split_label) % 10_000)

        if verbose:
            cls, cnt = np.unique(y[train_idx], return_counts=True)
            print(f"\n[{split_label}] Train dist BEFORE:", dict(zip(cls, cnt)))

        if equalize_classes:
            train_idx = balance_indices_equalize(
                y=y, idx=train_idx, rng=rng, method=equalize_method
            )
            if verbose:
                cls, cnt = np.unique(y[train_idx], return_counts=True)
                print(f"[{split_label}] Train dist AFTER:", dict(zip(cls, cnt)))

        docs_train = docs.iloc[train_idx]
        docs_test = docs.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if verbose:
            print(f"\n--- {split_label.upper()} ---")
            print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

        best_local: dict = {
            "min_df": None, "macro_f1": -1.0,
            "clf": None, "vec": None, "y_pred": None, "y_prob": None,
            "log_loss": None, "n_features": None,
        }

        for md in min_df_grid:
            # Use binary presence features for multiclass, matching the binary setup.
            vec = CountVectorizer(token_pattern=r"[^ ]+", binary=True, min_df=md)
            X_train_ng = vec.fit_transform(docs_train).astype(np.float32)
            X_test_ng = vec.transform(docs_test).astype(np.float32)
            X_train = X_train_ng
            X_test = X_test_ng
            if X_art_all is not None:
                X_train_art = scipy.sparse.csr_matrix(X_art_all[train_idx].astype(np.float32))
                X_test_art = scipy.sparse.csr_matrix(X_art_all[test_idx].astype(np.float32))
                X_train = scipy.sparse.hstack([X_train_ng, X_train_art], format="csr")
                X_test = scipy.sparse.hstack([X_test_ng, X_test_art], format="csr")
            if X_sem_all is not None:
                X_train_sem = X_sem_all[train_idx]
                X_test_sem = X_sem_all[test_idx]
                X_train = scipy.sparse.hstack([X_train, X_train_sem], format="csr")
                X_test = scipy.sparse.hstack([X_test, X_test_sem], format="csr")
            if X_pos_all is not None:
                X_train_pos = X_pos_all[train_idx]
                X_test_pos = X_pos_all[test_idx]
                X_train = scipy.sparse.hstack([X_train, X_train_pos], format="csr")
                X_test = scipy.sparse.hstack([X_test, X_test_pos], format="csr")


            clf = LogisticRegression(max_iter=4000, class_weight="balanced")

            if certainty_to_weight is not None and "certainty_level" in df_fam.columns:
                train_df_idx = idx_all[train_idx]
                sw = (
                    df_fam.loc[train_df_idx, "certainty_level"]
                    .map(certainty_to_weight)
                    .fillna(1.0)
                    .values)

                clf.fit(X_train, y_train, sample_weight=sw)
            else:

                clf.fit(X_train, y_train)


            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            macro = f1_score(y_test, y_pred, average="macro")
            loss = log_loss(y_test, y_prob, labels=class_labels)

            if verbose:
                print(
                    f"min_df={md}  features={X_train.shape[1]}  "
                    f"macro_f1={macro:.4f}  log_loss={loss:.4f}"
                )
                print(classification_report(y_test, y_pred, digits=4))

            if macro > best_local["macro_f1"]:
                best_local.update(
                    min_df=md, macro_f1=macro, clf=clf, vec=vec,
                    y_pred=y_pred, y_prob=y_prob, log_loss=loss,
                    n_features=X_train.shape[1],
                )

        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_test, best_local["y_pred"], average="macro", zero_division=0
        )
        p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
            y_test, best_local["y_pred"], labels=class_labels, average=None, zero_division=0
        )
        class_metric_map = {
            cls: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
            }
            for cls, p, r, f in zip(class_labels, p_cls, r_cls, f_cls)
        }
        eval_metrics = {
            "accuracy": float(accuracy_score(y_test, best_local["y_pred"])),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f_macro),
            "log_loss": float(best_local["log_loss"]),
        }
        for cls, vals in class_metric_map.items():
            safe = _safe_metric_label(cls)
            eval_metrics[f"precision_{safe}"] = vals["precision"]
            eval_metrics[f"recall_{safe}"] = vals["recall"]
            eval_metrics[f"f1_{safe}"] = vals["f1"]
        best_local["eval_metrics"] = eval_metrics
        best_local["class_metric_map"] = class_metric_map

        test_df_idx = idx_all[test_idx]
        df_eval = df_fam.loc[
            test_df_idx,
            ["lemma", "ipa", "origin_type", "primary_donor_family", "family_clean"],
        ].copy()
        df_eval["true_family"] = y_test
        df_eval["pred_family"] = best_local["y_pred"]
        df_eval["correct"] = df_eval["true_family"] == df_eval["pred_family"]
        for i, cls in enumerate(class_labels):
            df_eval[f"prob_{_safe_metric_label(cls)}"] = best_local["y_prob"][:, i]

        best_local["vowel_coef_map"] = extract_vowel_coefs(
            best_local["vec"], best_local["clf"]
        )
        return best_local, df_eval

    # Mode dispatch
    all_scores: list[float] = []
    all_losses: list[float] = []
    all_metric_rows: list[dict] = []
    all_eval_tables: list[pd.DataFrame] = []
    best_overall: dict | None = None
    best_eval: pd.DataFrame | None = None

    if mode == "single":
        tr, te = train_test_split(
            np.arange(len(df_fam)), test_size=test_size,
            stratify=y, random_state=random_state,
        )
        best_overall, best_eval = fit_one_split(tr, te, "single")
        summary = None

    elif mode == "repeated_holdout":
        for r in range(n_runs):
            tr, te = train_test_split(
                np.arange(len(df_fam)), test_size=test_size,
                stratify=y, random_state=random_state + r,
            )
            bl, de = fit_one_split(tr, te, f"run {r + 1}/{n_runs}")
            all_scores.append(bl["macro_f1"])
            all_losses.append(bl["log_loss"])
            all_metric_rows.append(bl["eval_metrics"])
            all_eval_tables.append(de.copy())
            all_vowel_maps.append(bl["vowel_coef_map"])
            if best_overall is None or bl["macro_f1"] > best_overall["macro_f1"]:
                best_overall, best_eval = bl, de

        scores = np.array(all_scores)
        losses = np.array(all_losses, dtype=float)
        metrics_df = pd.DataFrame(all_metric_rows)
        summary = {
            "mode": mode, "test_size": test_size, "n_runs": n_runs,
            "min_df_grid": list(min_df_grid),
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "macro_f1_min": float(scores.min()),
            "macro_f1_max": float(scores.max()),
            "log_loss_mean": float(losses.mean()),
            "log_loss_std": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
            "log_loss_min": float(losses.min()),
            "log_loss_max": float(losses.max()),
            "accuracy_mean": float(metrics_df["accuracy"].mean()),
            "precision_macro_mean": float(metrics_df["precision_macro"].mean()),
            "recall_macro_mean": float(metrics_df["recall_macro"].mean()),
            "f1_macro_mean": float(metrics_df["f1_macro"].mean()),
            "fold_evals": all_eval_tables,
            "vowel_coef_summary": summarize_coef_maps(all_vowel_maps),
            "semantic_feature_names": semantic_feature_names,
            "pos_feature_names": pos_feature_names,
        }
        class_metric_means: dict[str, dict[str, float]] = {}
        for cls in class_labels:
            safe = _safe_metric_label(cls)
            class_metric_means[str(cls)] = {
                "precision_mean": float(metrics_df[f"precision_{safe}"].mean()),
                "recall_mean": float(metrics_df[f"recall_{safe}"].mean()),
                "f1_mean": float(metrics_df[f"f1_{safe}"].mean()),
            }
            summary[f"precision_{safe}_mean"] = class_metric_means[str(cls)]["precision_mean"]
            summary[f"recall_{safe}_mean"] = class_metric_means[str(cls)]["recall_mean"]
            summary[f"f1_{safe}_mean"] = class_metric_means[str(cls)]["f1_mean"]
        summary["class_metric_means"] = class_metric_means
        summary["class_labels"] = [str(c) for c in class_labels]
        if verbose:
            print("\n=== REPEATED HOLDOUT SUMMARY ===")
            _print_summary(summary)

    elif mode == "kfold":
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        for fold, (tr, te) in enumerate(
            skf.split(np.zeros(len(df_fam)), y), start=1
        ):
            bl, de = fit_one_split(tr, te, f"fold {fold}/{k}")
            all_scores.append(bl["macro_f1"])
            all_losses.append(bl["log_loss"])
            all_metric_rows.append(bl["eval_metrics"])
            all_eval_tables.append(de.copy())
            all_vowel_maps.append(bl["vowel_coef_map"])
            if best_overall is None or bl["macro_f1"] > best_overall["macro_f1"]:
                best_overall, best_eval = bl, de

        scores = np.array(all_scores)
        losses = np.array(all_losses, dtype=float)
        metrics_df = pd.DataFrame(all_metric_rows)
        summary = {
            "mode": mode, "k": k,
            "min_df_grid": list(min_df_grid),
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "macro_f1_min": float(scores.min()),
            "macro_f1_max": float(scores.max()),
            "log_loss_mean": float(losses.mean()),
            "log_loss_std": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
            "log_loss_min": float(losses.min()),
            "log_loss_max": float(losses.max()),
            "accuracy_mean": float(metrics_df["accuracy"].mean()),
            "precision_macro_mean": float(metrics_df["precision_macro"].mean()),
            "recall_macro_mean": float(metrics_df["recall_macro"].mean()),
            "f1_macro_mean": float(metrics_df["f1_macro"].mean()),
            "fold_evals": all_eval_tables,
            "vowel_coef_summary": summarize_coef_maps(all_vowel_maps),
            "semantic_feature_names": semantic_feature_names,
            "pos_feature_names": pos_feature_names,
        }
        class_metric_means: dict[str, dict[str, float]] = {}
        for cls in class_labels:
            safe = _safe_metric_label(cls)
            class_metric_means[str(cls)] = {
                "precision_mean": float(metrics_df[f"precision_{safe}"].mean()),
                "recall_mean": float(metrics_df[f"recall_{safe}"].mean()),
                "f1_mean": float(metrics_df[f"f1_{safe}"].mean()),
            }
            summary[f"precision_{safe}_mean"] = class_metric_means[str(cls)]["precision_mean"]
            summary[f"recall_{safe}_mean"] = class_metric_means[str(cls)]["recall_mean"]
            summary[f"f1_{safe}_mean"] = class_metric_means[str(cls)]["f1_mean"]
        summary["class_metric_means"] = class_metric_means
        summary["class_labels"] = [str(c) for c in class_labels]
        if verbose:
            print("\n=== K-FOLD SUMMARY ===")
            _print_summary(summary)

    else:
        raise ValueError("mode must be 'single', 'repeated_holdout', or 'kfold'")

    return (
        best_overall["clf"],
        best_overall["vec"],
        best_eval,
        best_overall["min_df"],
        summary,
    )


# Internal display helper

def _print_summary(summary: dict) -> None:
    """Print macro-F1/log-loss stats and top vowel-feature coefficients."""
    if "accuracy_mean" in summary:
        print(f"  accuracy: {summary['accuracy_mean']:.4f}")
    print(
        f"  macro_f1: {summary['macro_f1_mean']:.4f} "
        f"± {summary['macro_f1_std']:.4f} "
        f"[{summary['macro_f1_min']:.4f}, {summary['macro_f1_max']:.4f}]"
    )
    if "log_loss_mean" in summary:
        print(
            f"  log_loss: {summary['log_loss_mean']:.4f} "
            f"± {summary['log_loss_std']:.4f} "
            f"[{summary['log_loss_min']:.4f}, {summary['log_loss_max']:.4f}]"
        )
    vcs = summary.get("vowel_coef_summary", {})
    if vcs:
        items = sorted(vcs.items(), key=lambda kv: abs(kv[1]["mean"]), reverse=True)
        print("\n  Vowel coef summary (top 20 by |mean|):")
        for f, s in items[:20]:
            print(
                f"    {f:22s}  mean={s['mean']:+.4f}  std={s['std']:.4f}  "
                f"[{s['min']:+.4f}, {s['max']:+.4f}]  n={s['n_present']}"
            )

