"""
etymology_analysis_tables.py
--> for statistics included
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PERIOD_BINS = [-np.inf, 1300, 1526, 1700, 1867, np.inf]
DEFAULT_PERIOD_LABELS = [
    "up to 1300",
    "1301\u20131526",
    "1527\u20131700",
    "1701\u20131867",
    "1868 and after"]

MISSING_ORIGIN_TYPE = "Missing origin type"

ORIGIN_TYPE_DISPLAY = {
    "loan": "Loanword",
    "internal": "Internal formation",
    "international": "International word",
    "fictional": "Fictional",
    "onomatopoetic": "Onomatopoetic",
    "inherited": "Inherited",
    "unknown": "Unknown origin",
    "wanderword": "Wanderword",
    "calque": "Calque",
    "loanblend": "Loanblend",
    "disputed": "Disputed",
    MISSING_ORIGIN_TYPE: "Missing"}
MISSING_FORMATION_TYPE = "Missing formation type"
MISSING_DONOR_FAMILY = "Missing donor family"
DISPUTED_DONOR_FAMILY = "Disputed donor family"


@dataclass(frozen=True) # --> IMMUTABLE
class ExperimentConfig:


    native_origin_types: set[str] = field(default_factory=lambda: {
        "inherited",
        "internal",
        "onomatopoetic",
        "fictional",
        ""})
    
    loan_origin_types: set[str] = field(default_factory=lambda: {
        "loan",
        "international",
        "wanderword"})
    
    exclude_origin_types: set[str] = field(default_factory=lambda: {
        "calque",
        "unknown",
        "loanblend"})
    
    exclude_formation_types: set[str] = field(default_factory=set)
    
    exclude_entry_types: set[str] = field(default_factory=lambda: {
        "prefix_family",
        "suffix_family",
        "roman_grouped_tilde_family"})
    
    include_ambiguous: bool = False
    include_dialectal: bool = True
    include_archaic: bool = True
    include_obsolete: bool = True
    exclude_if_any_excluded_origin: bool = True
    display_labels: Mapping[str, str] = field(default_factory=dict)

    def with_updates(self, **updates) -> "ExperimentConfig":

        return replace(self, **updates)


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


def prepare_for_tables(df: pd.DataFrame, *,
    year_col: str = "first_year",
    period_col: str = "period",
    period_bins: list[float] | None = None,
    period_labels: list[str] | None = None) -> pd.DataFrame:
    """Return a copy with numeric year and period columns."""
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()

    if year_col not in out.columns:
        raise ValueError(f"Missing year column: {year_col}")

    period_bins = period_bins or DEFAULT_PERIOD_BINS
    period_labels = period_labels or DEFAULT_PERIOD_LABELS

    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    n_dropped = int(out[year_col].isna().sum())
    if n_dropped:
        print(f"Note: {n_dropped} rows dropped (missing {year_col!r})")
    out = out[out[year_col].notna()].copy()
    out[period_col] = pd.cut(
        out[year_col],
        bins=period_bins,
        labels=period_labels,
        right=True,
    )
    return out


def make_count_table(df: pd.DataFrame, *,
    group_col: str,
    period_col: str = "period",
    value_col: str = "lemma",
    period_labels: Iterable[str] | None = None,
    missing_label: str = "Missing",
    order: Iterable[str] | None = None,
    sort_by_total: bool = True,
    add_total_row: bool = True) -> pd.DataFrame:
    """Count rows by one grouping column and period."""
    period_labels = list(period_labels or DEFAULT_PERIOD_LABELS)
    work = df.copy()
    work[group_col] = clean_label_column(work, group_col, missing_label=missing_label)

    table = pd.pivot_table(
        work,
        index=group_col,
        columns=period_col,
        values=value_col,
        aggfunc="count",
        fill_value=0,
        observed=False)
    table = table.reindex(columns=period_labels, fill_value=0)
    table["Total"] = table.sum(axis=1)

    if order is not None:
        order = [value for value in order if value in table.index]
        remaining = sorted(value for value in table.index if value not in set(order))
        table = table.reindex(order + remaining, fill_value=0)
    elif sort_by_total:
        table = table.sort_values("Total", ascending=False)

    if add_total_row:
        table = pd.concat([table, pd.DataFrame([table.sum(axis=0)], index=["Total"])])

    return table


def donor_family_table(df: pd.DataFrame, *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    exclude_structural_origins: bool = True,
    donor_family_col: str = "primary_donor_family",
    collapse_disputed_families: bool = True) -> pd.DataFrame:
    """Build the donor-family-by-period table without hard-coded family remapping."""
    work = df.copy()
    if exclude_structural_origins:
        work = work[~pipe_contains_any(work.get("origin_type", ""), config.exclude_origin_types)].copy()
    work[donor_family_col] = clean_label_column(
        work,
        donor_family_col,
        missing_label=MISSING_DONOR_FAMILY).map(
        lambda value: normalize_donor_family_label(
            value,
            config=config,
            collapse_disputed_families=collapse_disputed_families))
    return make_count_table(
        work,
        group_col=donor_family_col,
        missing_label=MISSING_DONOR_FAMILY)


def origin_type_table(
    df: pd.DataFrame,
    *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    origin_col: str = "origin_type",
) -> pd.DataFrame:
    """Build the origin-type-by-period table using parser origin_type values."""
    work = df.copy()
    work[origin_col] = clean_label_column(
        work,
        origin_col,
        missing_label=MISSING_ORIGIN_TYPE,
    ).map(lambda value: display_label(value, config))
    return make_count_table(
        work,
        group_col=origin_col,
        missing_label=MISSING_ORIGIN_TYPE,
    )


def build_period_summary_tables(
    df: pd.DataFrame,
    *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    period_bins: list[float] | None = None,
    period_labels: list[str] | None = None,
    exclude_structural_origins: bool = True,
    collapse_disputed_families: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare data and return donor-family and origin-type period tables.
    Returns '(prepared_df, family_table, type_table)'.
    """
    prepared = prepare_for_tables(df,
        period_bins=period_bins,
        period_labels=period_labels)
    family = donor_family_table(prepared,
        config=config,
        exclude_structural_origins=exclude_structural_origins,
        collapse_disputed_families=collapse_disputed_families)
    origin = origin_type_table(prepared, config=config)
    return prepared, family, origin


def plot_origin_by_period(df: pd.DataFrame, *,
    origin_types: Iterable[str] = ("loan", "inherited"),
    formation_type: str = "simplex",
    target_families: Iterable[str] = (
        "Uralic",
        "Slavic",
        "German",
        "Turkic",
        "Latin",
        "Romance",
    ),
    year_col: str = "first_year",
    origin_col: str = "origin_type",
    formation_col: str = "formation_type",
    donor_family_col: str = "primary_donor_family",
    period_bins: list[float] | None = None,
    period_labels: list[str] | None = None,
    figsize: tuple[int, int] = (16, 10),
    font_size: int = 22,
    title: str | None = None,
    save_path: str | None = None) -> tuple[pd.DataFrame, object, object]:
    """
    Plot lemma counts by donor family and first-attestation period.
    """

    period_bins = period_bins or DEFAULT_PERIOD_BINS
    period_labels = period_labels or DEFAULT_PERIOD_LABELS
    target_families = list(target_families)
    origin_types = set(origin_types)

    work = df.copy()
    work.columns = work.columns.astype(str).str.strip()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")

    mask = (
        work[origin_col].isin(origin_types)
        & (work[formation_col] == formation_type)
        & work[donor_family_col].isin(target_families)
        & work[year_col].notna()
    )
    work = work[mask].copy()
    work["period"] = pd.cut(
        work[year_col],
        bins=period_bins,
        labels=period_labels,
        right=True,
    )

    table = (
        work.groupby([donor_family_col, "period"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=target_families, columns=period_labels, fill_value=0)
    )
    plot_df = table.loc[table.sum(axis=1) > 0].copy()

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 4,
        "axes.labelsize": font_size + 2,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "legend.title_fontsize": font_size,
    })

    fig, ax = plt.subplots(figsize=figsize)
    for row_name in plot_df.index:
        ax.plot(
            plot_df.columns,
            plot_df.loc[row_name].values,
            marker="o",
            markersize=10,
            linewidth=3.5,
            label=row_name,
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel("Period of first attestation")
    ax.set_ylabel("Number of lemmas")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Donor family", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return table, fig, ax


def emmorph_pos_summary(df: pd.DataFrame, *,
    lemma_col: str = "lemma",
    pos_col: str = "pos",
    archaic_col: str = "is_archaic",
    dialectal_col: str = "is_dialectal",
    obsolete_col: str = "is_obsolete") -> tuple[pd.DataFrame, pd.DataFrame]:


    work = df.copy()
    work.columns = work.columns.astype(str).str.strip()

    if lemma_col not in work.columns:
        raise ValueError(f"Missing lemma column: {lemma_col}")
    if pos_col not in work.columns:
        raise ValueError(f"Missing POS column: {pos_col}")

    work[lemma_col] = work[lemma_col].fillna("").astype(str).str.strip()
    work = work[work[lemma_col] != ""].copy()

    grouped = (work.groupby(lemma_col, as_index=False, sort=True)
            .agg({pos_col: lambda s: next(
            (value for value in s.fillna("").astype(str).str.strip() if value), ""),
            archaic_col: lambda s: any(as_bool(v) for v in s) if archaic_col in work.columns else False,
            dialectal_col: lambda s: any(as_bool(v) for v in s) if dialectal_col in work.columns else False,
            obsolete_col: lambda s: any(as_bool(v) for v in s) if obsolete_col in work.columns else False}))

    has_pos = grouped[pos_col].fillna("").astype(str).str.strip() != ""
    pos_subset = grouped[has_pos].copy()

    total_lemmas = len(grouped)
    lemmas_with_pos = int(has_pos.sum())
    lemmas_without_pos = int((~has_pos).sum())

    summary_table = pd.DataFrame(
        [
            {
                "metric": "total_lemmas",
                "count": total_lemmas,
                "percent": 100.0 if total_lemmas else np.nan},
            {
                "metric": "lemmas_with_emmorph_pos",
                "count": lemmas_with_pos,
                "percent": (100.0 * lemmas_with_pos / total_lemmas) if total_lemmas else np.nan},
            {
                "metric": "lemmas_without_emmorph_pos",
                "count": lemmas_without_pos,
                "percent": (100.0 * lemmas_without_pos / total_lemmas) if total_lemmas else np.nan}])

    flag_specs = [
        ("archaic", archaic_col),
        ("dialectal", dialectal_col),
        ("obsolete", obsolete_col),
    ]
    flag_rows: list[dict[str, object]] = []
    for label, column in flag_specs:
        if column in pos_subset.columns:
            count = int(pos_subset[column].map(as_bool).sum())
        else:
            count = 0
        flag_rows.append({
            "flag": label,
            "count": count,
            "percent_of_pos_tagged": (100.0 * count / lemmas_with_pos) if lemmas_with_pos else np.nan,
        })

    any_flag = pd.Series(False, index=pos_subset.index)
    for _, column in flag_specs:
        if column in pos_subset.columns:
            any_flag = any_flag | pos_subset[column].map(as_bool)
    flag_rows.append({
        "flag": "any_flag",
        "count": int(any_flag.sum()),
        "percent_of_pos_tagged": (100.0 * int(any_flag.sum()) / lemmas_with_pos) if lemmas_with_pos else np.nan,
    })

    flagged_table = pd.DataFrame(flag_rows)
    return summary_table, flagged_table


def emmorph_pos_flagged_table(df: pd.DataFrame, *,
    lemma_col: str = "lemma",
    pos_col: str = "pos",
    archaic_col: str = "is_archaic",
    dialectal_col: str = "is_dialectal",
    obsolete_col: str = "is_obsolete",
    line_break: str = "<br>") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    work = df.copy()
    work.columns = work.columns.astype(str).str.strip()

    if lemma_col not in work.columns:
        raise ValueError(f"Missing lemma column: {lemma_col}")
    if pos_col not in work.columns:
        raise ValueError(f"Missing POS column: {pos_col}")

    work[lemma_col] = work[lemma_col].fillna("").astype(str).str.strip()
    work = work[work[lemma_col] != ""].copy()

    agg_spec: dict[str, object] = {
        pos_col: lambda s: any(value for value in s.fillna("").astype(str).str.strip()),
    }
    for column in [archaic_col, dialectal_col, obsolete_col]:
        if column in work.columns:
            agg_spec[column] = lambda s: any(as_bool(v) for v in s)

    grouped = work.groupby(lemma_col, as_index=False, sort=True).agg(agg_spec)
    grouped["has_pos"] = grouped[pos_col].map(as_bool)

    flag_columns = [column for column in [archaic_col, dialectal_col, obsolete_col] if column in grouped.columns]
    if flag_columns:
        grouped["flagged"] = False
        for column in flag_columns:
            grouped["flagged"] = grouped["flagged"] | grouped[column].map(as_bool)
    else:
        grouped["flagged"] = False

    found_and_not_flagged = int((grouped["has_pos"] & ~grouped["flagged"]).sum())
    found_and_flagged = int((grouped["has_pos"] & grouped["flagged"]).sum())
    not_found_and_not_flagged = int((~grouped["has_pos"] & ~grouped["flagged"]).sum())
    not_found_and_flagged = int((~grouped["has_pos"] & grouped["flagged"]).sum())

    count_table = pd.DataFrame(
        [
            {
                "row": "found_pos",
                "not_flagged": found_and_not_flagged,
                "flagged": found_and_flagged,
            },
            {
                "row": "not_found_pos",
                "not_flagged": not_found_and_not_flagged,
                "flagged": not_found_and_flagged,
            },
        ]
    ).set_index("row")
    count_table["total"] = count_table.sum(axis=1)
    count_table.loc["total"] = count_table.sum(axis=0)

    percent_table = count_table.copy().astype(float)
    display_table = count_table.copy().astype(object)
    for column in count_table.columns:
        column_total = int(count_table.loc["total", column])
        for row in count_table.index:
            count = int(count_table.loc[row, column])
            pct = (100.0 * count / column_total) if column_total else np.nan
            percent_table.loc[row, column] = pct
            display_table.loc[row, column] = f"{count}{line_break}({pct:.1f}%)"

    return count_table, percent_table, display_table


def origin_formation_crosstab(df: pd.DataFrame, *,
    origin_col: str = "origin_type",
    formation_col: str = "formation_type",
    lemma_col: str = "lemma",
    drop_disputed_origin: bool = True,
    drop_disputed_formation: bool = True,
    drop_missing_origin: bool = False,
    drop_missing_formation: bool = False,
    add_totals: bool = True) -> pd.DataFrame:


    work = df.copy()
    work.columns = work.columns.astype(str).str.strip()

    if origin_col not in work.columns:
        raise ValueError(f"Missing origin column: {origin_col}")
    if formation_col not in work.columns:
        raise ValueError(f"Missing formation column: {formation_col}")

    if lemma_col in work.columns:
        work = work[work[lemma_col].notna()].copy()

    work[origin_col] = work[origin_col].fillna("").astype(str).str.strip()
    work[formation_col] = work[formation_col].fillna("").astype(str).str.strip()

    if drop_disputed_origin:
        work = work[~work[origin_col].str.contains(r"\|", regex=True, na=False)].copy()
    if drop_disputed_formation:
        work = work[~work[formation_col].str.contains(r"\|", regex=True, na=False)].copy()

    if drop_missing_origin:
        work = work[work[origin_col] != ""].copy()
    else:
        work.loc[work[origin_col] == "", origin_col] = MISSING_ORIGIN_TYPE

    if drop_missing_formation:
        work = work[work[formation_col] != ""].copy()
    else:
        work.loc[work[formation_col] == "", formation_col] = MISSING_FORMATION_TYPE

    table = pd.crosstab(
        index=work[formation_col],
        columns=work[origin_col],
        dropna=False,
    )

    table.index.name = formation_col
    table.columns.name = origin_col

    if add_totals:
        table["Total"] = table.sum(axis=1)
        total_row = pd.DataFrame([table.sum(axis=0)], index=["Total"])
        table = pd.concat([table, total_row])

    return table


_LOAN_LABELS = {"loan", "international", "wanderword"}
_NATIVE_LABELS = {"inherited", "internal", "onomatopoetic", "fictional", "coinage"}

_ENTRY_TYPE_DISPLAY = {
    "single": "Single entry",
    "family": "Word family",
    "suffix_family": "Suffix family",
    "prefix_family": "Prefix family",
    "roman_grouped_tilde_family": "Roman-grouped family",
}


def _origin_group(value: str) -> str:
    parts = {p.strip() for p in str(value).split("|") if p.strip()}
    if parts & _LOAN_LABELS:
        return "Loan"
    if parts & _NATIVE_LABELS:
        return "Native"
    if "unknown" in parts:
        return "Unknown"
    return "Other/Missing"


def add_experiment_labels(df: pd.DataFrame, *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    output_col: str = "experiment_class",
) -> pd.DataFrame:
    """Add native/loan/exclude/ambiguous labels according to a config."""
    out = df.copy()
    out[output_col] = [
        classify_origin_type(value, config=config)
        for value in out.get("origin_type", pd.Series("", index=out.index))
    ]

    if "formation_type" in out.columns and config.exclude_formation_types:
        out.loc[
            pipe_contains_any(out["formation_type"], config.exclude_formation_types),
            output_col,
        ] = "exclude"

    if "entry_type" in out.columns and config.exclude_entry_types:
        out.loc[
            pipe_contains_any(out["entry_type"], config.exclude_entry_types),
            output_col,
        ] = "exclude"

    for column, include in [
        ("is_dialectal", config.include_dialectal),
        ("is_archaic", config.include_archaic),
        ("is_obsolete", config.include_obsolete),
    ]:
        if column in out.columns and not include:
            out.loc[out[column].map(as_bool), output_col] = "exclude"

    if not config.include_ambiguous:
        out.loc[out[output_col] == "ambiguous", output_col] = "exclude"

    return out


def experiment_subset(
    df: pd.DataFrame,
    *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    classes: Iterable[str] = ("native", "loan"),
) -> pd.DataFrame:
    """Return rows included in an experiment according to configurable classes."""
    out = add_experiment_labels(df, config=config)
    wanted = set(classes)
    return out[out["experiment_class"].isin(wanted)].copy()


def classify_origin_type(
    value: object,
    *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> str:
    """Classify a possibly pipe-separated origin_type into native/loan/etc."""
    parts = split_pipe(value)
    if not parts:
        return "native" if "" in config.native_origin_types else "exclude"

    native_hits = bool(parts & config.native_origin_types)
    loan_hits = bool(parts & config.loan_origin_types)
    excluded_hits = bool(parts & config.exclude_origin_types)

    if excluded_hits and config.exclude_if_any_excluded_origin:
        return "exclude"
    if loan_hits and native_hits:
        return "ambiguous"
    if loan_hits:
        return "loan"
    if native_hits:
        return "native"
    if excluded_hits:
        return "exclude"
    return "other"


def split_pipe(value: object) -> set[str]:
    """Return normalized parts from a pipe-separated field."""
    if pd.isna(value):
        return set()
    return {
        part.strip()
        for part in str(value).split("|")
        if part.strip()
    }


def pipe_contains_any(values: object, wanted: Iterable[str]) -> pd.Series:
    """Vectorized test for whether pipe-separated cells contain any wanted value."""
    wanted = {str(value).strip() for value in wanted}
    if isinstance(values, pd.Series):
        return values.fillna("").map(lambda value: bool(split_pipe(value) & wanted))
    return pd.Series([bool(split_pipe(values) & wanted)])


def clean_label_column(
    df: pd.DataFrame,
    column: str,
    *,
    missing_label: str,
) -> pd.Series:
    """Return a stripped string column with empty cells replaced by missing_label."""
    if column in df.columns:
        values = df[column]
    else:
        values = pd.Series("", index=df.index)
    cleaned = values.fillna("").astype(str).str.strip()
    return cleaned.replace("", missing_label)


def display_label(value: str, config: ExperimentConfig) -> str:
    """Apply caller-provided display labels to single or pipe-separated values."""
    parts = [part.strip() for part in str(value).split("|") if part.strip()]
    if not parts:
        return value
    return "|".join(config.display_labels.get(part, part) for part in parts)


def normalize_donor_family_label(
    value: str,
    *,
    config: ExperimentConfig,
    collapse_disputed_families: bool = True,
) -> str:
    """Normalize donor-family labels for display tables."""
    if collapse_disputed_families and "|" in str(value):
        return DISPUTED_DONOR_FAMILY
    return display_label(value, config)


def as_bool(value: object) -> bool:
    """Interpret yes/no and true/false style cells."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return True
    if text in {"no", "false", "0", ""}:
        return False
    return bool(value)


def entry_type_stats_table(
    df: pd.DataFrame,
    *,
    entry_type_col: str = "entry_type",
    origin_col: str = "origin_type",
) -> pd.DataFrame:

    work = df.copy()
    work[origin_col] = work[origin_col].fillna("").astype(str).str.strip()
    work["_origin_group"] = work[origin_col].apply(_origin_group)
    work[entry_type_col] = work[entry_type_col].fillna("").astype(str).str.strip()

    origin_cols = ["Loan", "Native", "Unknown", "Other/Missing"]
    table = (
        pd.crosstab(work[entry_type_col], work["_origin_group"])
        .reindex(columns=origin_cols, fill_value=0)
    )

    table.index = [
        _ENTRY_TYPE_DISPLAY.get(str(value), str(value))
        for value in table.index
    ]
    table["Total"] = table.sum(axis=1)
    total_all = int(table["Total"].sum())
    table["Percent"] = (
        table["Total"] / total_all * 100
        if total_all
        else np.nan
    )

    table = table.sort_values("Total", ascending=False)
    total_row = table[origin_cols + ["Total"]].sum(axis=0)
    total_row["Percent"] = 100.0 if total_all else np.nan
    table.loc["Total"] = total_row

    return table


def compact_period_table(
    table: pd.DataFrame,
    *,
    period_labels: list[str],
    row_renames: dict[str, str] | None = None,
    top_n: int | None = None,
    other_label: str = "Other",
    add_total_row: bool = True,
) -> pd.DataFrame:

    renames = row_renames or {}
    data = table.drop(index="Total", errors="ignore").copy()

    if "Total" not in data.columns:
        data["Total"] = data.sum(axis=1)

    data = data.sort_values("Total", ascending=False)

    if top_n is not None and len(data) > top_n:
        top = data.iloc[:top_n]
        other = data.iloc[top_n:].sum(axis=0).to_frame().T
        other.index = [other_label]
        data = pd.concat([top, other])

    data.index = [renames.get(str(idx), str(idx)) for idx in data.index]

    col_order = [col for col in period_labels if col in data.columns]
    if "Total" in data.columns:
        col_order.append("Total")
    data = data.reindex(columns=col_order, fill_value=0)

    if add_total_row:
        data.loc["Total"] = data.sum(axis=0)

    return data


def donor_family_period_table(
    df: pd.DataFrame,
    *,
    config: ExperimentConfig | None = None,
    loan_only: bool = True,
    loan_origin_types: set[str] | None = None,
    formation_type: str | None = None,
    top_n: int | None = 10,
    period_bins: list[float] | None = None,
    period_labels: list[str] | None = None,
) -> pd.DataFrame:

    period_bins = period_bins or DEFAULT_PERIOD_BINS
    period_labels = period_labels or DEFAULT_PERIOD_LABELS
    active_config = config or DEFAULT_EXPERIMENT_CONFIG

    work = df.copy()

    if loan_only:
        loan_labels = loan_origin_types or {"loan"}
        work = work[
            work["origin_type"].fillna("").astype(str).apply(
                lambda value: bool(split_pipe(value) & loan_labels)
            )
        ].copy()

    if formation_type is not None:
        work = work[
            work.get("formation_type", "")
            .fillna("")
            .astype(str)
            .str.strip()
            == formation_type
        ].copy()

    prepared = prepare_for_tables(
        work,
        period_bins=period_bins,
        period_labels=period_labels,
    )
    table = donor_family_table(prepared, config=active_config)

    return compact_period_table(
        table,
        period_labels=period_labels,
        top_n=top_n,
    )


def origin_type_period_table(
    df: pd.DataFrame,
    *,
    config: ExperimentConfig | None = None,
    exclude_pipe_separated: bool = True,
    formation_type: str | None = None,
    period_bins: list[float] | None = None,
    period_labels: list[str] | None = None,
) -> pd.DataFrame:

    period_bins = period_bins or DEFAULT_PERIOD_BINS
    period_labels = period_labels or DEFAULT_PERIOD_LABELS
    active_config = config or DEFAULT_EXPERIMENT_CONFIG

    prepared = prepare_for_tables(
        df,
        period_bins=period_bins,
        period_labels=period_labels,
    )

    if formation_type is not None:
        prepared = prepared[
            prepared.get("formation_type", "")
            .fillna("")
            .astype(str)
            .str.strip()
            == formation_type
        ].copy()

    if exclude_pipe_separated:
        prepared = prepared[
            ~prepared["origin_type"]
            .fillna("")
            .astype(str)
            .str.contains(r"\|", regex=True)
        ].copy()

    table = origin_type_table(prepared, config=active_config)

    return compact_period_table(
        table,
        period_labels=period_labels,
        row_renames=ORIGIN_TYPE_DISPLAY,
        top_n=None,
    )


def origin_formation_crosstab_table(
    df: pd.DataFrame,
    *,
    origin_col: str = "origin_type",
    formation_col: str = "formation_type",
    drop_disputed_origin: bool = True,
    drop_disputed_formation: bool = True,
    column_renames: dict[str, str] | None = None,
    row_renames: dict[str, str] | None = None,
) -> pd.DataFrame:

    table = origin_formation_crosstab(
        df,
        origin_col=origin_col,
        formation_col=formation_col,
        drop_disputed_origin=drop_disputed_origin,
        drop_disputed_formation=drop_disputed_formation,
        add_totals=True,
    )

    col_renames = {
        "Missing origin type": "Missing",
        "Missing formation type": "Missing",
    }
    if column_renames:
        col_renames.update(column_renames)

    row_renames_final = {
        "Missing formation type": "Missing",
        "constituent_lexicalization": "constituent lex.",
    }
    if row_renames:
        row_renames_final.update(row_renames)

    table = table.rename(columns=col_renames, index=row_renames_final)
    table.index = [str(value).replace("_", " ") for value in table.index]

    return table


DEFAULT_VERB_ENDING_PATTERNS: dict[str, tuple[str, ...]] = {
    "-ik": ("ik",),
    "-ál": ("ál",),
    "-og/-eg/-ög": ("og", "eg", "ög"),
    "-an/-en": ("an", "en"),
    "-ad/-ed": ("ad", "ed"),
    "-z/-az/-oz/-ez/-öz": ("z", "az", "oz", "ez", "öz"),
    "-al/-el/-él/-ol/-öl/-ul/-ül": ("al", "el", "él", "ol", "öl", "ul", "ül"),
    "-at/-et": ("at", "et"),
}


def clean_binary_origin(value: object) -> str | None:

    text = "" if pd.isna(value) else str(value).strip().lower()
    if not text or "|" in text:
        return None
    if text in {"loan", "international", "wanderword"}:
        return "loan"
    if text in {"internal", "inherited", "onomatopoetic", "fictional"}:
        return "non-loan"
    return None


def verb_ending_origin_table(
    df: pd.DataFrame,
    *,
    lemma_col: str = "lemma",
    pos_col: str = "pos",
    origin_col: str = "origin_type",
    ending_patterns: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:

    patterns = ending_patterns or DEFAULT_VERB_ENDING_PATTERNS

    work = df.copy()
    for column in (lemma_col, pos_col, origin_col):
        if column not in work.columns:
            raise ValueError(f"Missing required column: {column}")

    verbs = work[
        work[pos_col].fillna("").astype(str).str.contains("VERB", na=False)
    ].copy()

    verbs["origin_binary"] = verbs[origin_col].apply(clean_binary_origin)
    verbs = verbs[verbs["origin_binary"].notna()].copy()

    verbs["_lemma_clean"] = (
        verbs[lemma_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for label, suffixes in patterns.items():
        verbs[label] = verbs["_lemma_clean"].str.endswith(suffixes, na=False)

    ending_cols = list(patterns.keys())

    return (
        verbs
        .groupby("origin_binary")
        .agg(
            n_verbs=(lemma_col, "size"),
            **{col: (col, lambda s: s.mean() * 100) for col in ending_cols},
        )
        .reindex(["non-loan", "loan"])
    )

def plot_verb_ending_origin_rates(
    ending_rates: pd.DataFrame,
    *,
    ending_patterns: dict[str, tuple[str, ...]] | None = None,
    figsize: tuple[int, int] = (11, 7),
) -> tuple[object, object]:
    """Plot verb-ending percentages for loan and non-loan verbs."""
    patterns = ending_patterns or DEFAULT_VERB_ENDING_PATTERNS
    ending_cols = list(patterns.keys())
    plot_data = ending_rates[ending_cols].T

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(plot_data.index))
    height = 0.36

    ax.barh(y + height / 2, plot_data["non-loan"], height, label="non-loan", color="#4C78A8")
    ax.barh(y - height / 2, plot_data["loan"], height, label="loan", color="#F58518")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_data.index, fontsize=14)
    ax.set_xlabel("Percentage of verbs within origin class", fontsize=14)
    ax.set_ylabel("Verb ending group", fontsize=12)

    ax.legend(title="Origin class", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_xlim(0, max(plot_data.max()) + 10)

    for i, suffix_group in enumerate(plot_data.index):
        for origin, offset in [("non-loan", height / 2), ("loan", -height / 2)]:
            value = plot_data.loc[suffix_group, origin]
            ax.text(value + 0.5, i + offset, f"{value:.1f}", va="center", ha="left", fontsize=12)

    fig.tight_layout(rect=[0, 0, 0.76, 1])
    return fig, ax

def semantic_label_by_donor_group_tables(
    df: pd.DataFrame,
    *,
    semantic_col: str = "semantic_field_odenet",
    origin_col: str = "origin_type",
    donor_family_col: str = "primary_donor_family",
    donor_language_col: str = "primary_donor_language",
    top_n: int | None = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return count and column-percentage tables for semantic labels by donor group."""

    group_order = [
        "Uralic",
        "other native",
        "German",
        "Latin",
        "Turkic",
        "Slavic",
        "Romance",
        "English",
        "international/wanderword",
        "other loan",
        "rest",
    ]

    def simplify_semantic_label(value: object) -> str:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return "MISSING"
        if text == "MISSING":
            return "MISSING"
        if text.startswith("noun."):
            return text.removeprefix("noun.")
        if text.startswith("verb."):
            return text.removeprefix("verb.")
        return text

    def split_pipe(value: object) -> set[str]:
        return {part.strip() for part in str(value).split("|") if part.strip()}

    def donor_group(row: pd.Series) -> str:
        origin_parts = split_pipe(row.get(origin_col, ""))
        family_parts = split_pipe(row.get(donor_family_col, ""))
        language_parts = split_pipe(row.get(donor_language_col, ""))

        if "Uralic" in family_parts:
            return "Uralic"
        if origin_parts & {"inherited", "internal", "onomatopoetic", "fictional"}:
            return "other native"
        if "German" in family_parts:
            return "German"
        if "Latin" in family_parts:
            return "Latin"
        if "Turkic" in family_parts:
            return "Turkic"
        if "Slavic" in family_parts:
            return "Slavic"
        if "Romance" in family_parts:
            return "Romance"
        if "English" in language_parts:
            return "English"
        if origin_parts & {"international", "wanderword"}:
            return "international/wanderword"
        if "loan" in origin_parts:
            return "other loan"
        return "rest"

    work = df.copy()
    work["_semantic_label_simple"] = work[semantic_col].map(simplify_semantic_label)
    work["_donor_group"] = work.apply(donor_group, axis=1)

    full_counts = pd.crosstab(
        work["_semantic_label_simple"],
        work["_donor_group"],
    ).reindex(columns=group_order, fill_value=0)

    full_counts["Total"] = full_counts.sum(axis=1)
    full_counts = full_counts.sort_values("Total", ascending=False)

    column_totals = full_counts[group_order].sum(axis=0)

    if top_n is not None:
        counts = full_counts.head(top_n).copy()
    else:
        counts = full_counts.copy()

    counts.loc["Total"] = list(column_totals) + [int(full_counts["Total"].sum())]

    percentages = (
        counts[group_order]
        .drop(index="Total", errors="ignore")
        .div(column_totals.replace(0, pd.NA), axis=1)
        .mul(100)
        .round(1)
    )

    return counts, percentages