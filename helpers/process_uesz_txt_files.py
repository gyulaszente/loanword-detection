from pathlib import Path
import json
import re
import time
import importlib
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict

from . import normalization as nm
from . import entry_predicates as ep
from . import etymology_parser as etp

importlib.reload(nm)
importlib.reload(ep)
importlib.reload(etp)

extract_entry_block = etp.extract_entry_block


ARROW_REF_RE = re.compile(r"→\s*([^\s,.;:!?()[\]{}<>|]+)")

_HOMOGRAPH_RE = re.compile(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+$')

_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _strip_homograph(lemma: str) -> str:
    return _HOMOGRAPH_RE.sub('', lemma).strip()


def _lemma_with_superscript(lemma: str, homograph: str | int | None) -> str | None:
    """Return e.g. 'nap¹' for lemma='nap', homograph='1' (or 1). None if no homograph."""
    if not homograph:
        return None
    sup = str(homograph).translate(_SUPERSCRIPT_DIGITS)
    return f"{lemma}{sup}"


_FORMATION_KEYS = [
    "formation_type",
    "derivation_base",
    "derivation_base_origin",
    "compound_parts",
    "compound_parts_origins",
]


def _reorder_formation_fields(d: dict) -> dict:
    """Move formation_type and related fields to immediately after origin_type."""
    formation_vals = {k: d[k] for k in _FORMATION_KEYS if k in d}
    if not formation_vals or "origin_type" not in d:
        return d
    new_d = {}
    for k, v in d.items():
        if k in _FORMATION_KEYS:
            continue
        new_d[k] = v
        if k == "origin_type":
            for fk in _FORMATION_KEYS:
                if fk in formation_vals:
                    new_d[fk] = formation_vals[fk]
    return new_d


def _build_lemma_index(json_root: Path) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(json_root.rglob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entry["_path"] = str(path)
        lemma = entry.get("lemma") or path.stem
        index[lemma].append(entry)
        plain = _strip_homograph(lemma)
        if plain != lemma:
            index[plain].append(entry)
        # Also index under lemma+superscript so 'nap¹' finds only homograph 1
        with_sup = _lemma_with_superscript(plain, entry.get("homograph"))
        if with_sup and with_sup != lemma:
            index[with_sup].append(entry)
        # Index subentries by their own lemma, origin_type lives there in family entries
        for sub in entry.get("subentries") or []:
            sub_lemma = sub.get("lemma")
            if not sub_lemma:
                continue
            sub["_path"] = str(path)
            index[sub_lemma].append(sub)
            plain_sub = _strip_homograph(sub_lemma)
            if plain_sub != sub_lemma:
                index[plain_sub].append(sub)
            with_sup_sub = _lemma_with_superscript(plain_sub, sub.get("homograph"))
            if with_sup_sub and with_sup_sub != sub_lemma:
                index[with_sup_sub].append(sub)
    return dict(index)


def _entry_origin(entry: dict) -> str | None:
    family = entry.get("primary_donor_family")
    if family:
        return family
    origin_type = entry.get("origin_type")
    if origin_type == "inherited":
        return "Uralic"
    # covers onomatopoetic, loan, internal, unknown.
    return origin_type if origin_type else None


def _is_informative_origin(origin: str | None) -> bool:
    return origin not in {None, "", "internal", "unknown", "uncertain"}


def _resolve_compound_origin(
    compound_parts: list[str] | None,
    current_path: str,
    index: dict[str, list[dict]],
    visited: set[str],
    max_depth: int,
) -> str | None:
    if not compound_parts or max_depth == 0:
        return None
    informative: list[str] = []
    for part in compound_parts:
        origin = _resolve_base_origin(part, current_path, index, visited, max_depth - 1)
        if _is_informative_origin(origin):
            informative.append(origin)
    if not informative:
        return None
    unique = list(dict.fromkeys(informative))
    if len(unique) == 1:
        return unique[0]
    return unique[0]


def _resolve_base_origin_from_siblings(base_lemma: str, entry: dict) -> str | None:
    """Resolve a base from another subentry in the same family file."""
    if not base_lemma:
        return None
    base_plain = _strip_homograph(base_lemma)
    for sub in entry.get("subentries") or []:
        lemma = sub.get("lemma")
        if not lemma:
            continue
        lemma_plain = _strip_homograph(lemma)
        lemma_with_sup = _lemma_with_superscript(lemma_plain, sub.get("homograph"))
        if base_lemma not in {lemma, lemma_plain, lemma_with_sup} and base_plain != lemma_plain:
            continue
        origin = _entry_origin(sub)
        if origin in {"onomatopoetic", "fictional"} or _is_informative_origin(origin):
            return origin
    return None


def _resolve_base_origin(base_lemma: str,
    current_path: str,
    index: dict[str, list[dict]],
    visited: set[str],
    max_depth: int = 6) -> str | None:
    if not base_lemma or max_depth == 0:
        return None

    candidates = [
        e for e in index.get(base_lemma, [])
        if e.get("_path") != current_path]
    if not candidates:
        return None
    entry = candidates[0]
    entry_path = entry.get("_path", base_lemma)
    if entry_path in visited:
        return None
    visited.add(entry_path)

    direct_origin = _entry_origin(entry)
    if direct_origin in {"onomatopoetic", "fictional"}:
        return direct_origin
    if _is_informative_origin(direct_origin):
        return direct_origin

    if entry.get("derivation_base") and (
        entry.get("formation_type") == "derivation"
        or entry.get("origin_type") == "internal"):
        deeper = _resolve_base_origin(
            entry["derivation_base"], entry_path, index, visited, max_depth - 1)
        if deeper:
            return deeper
    if entry.get("compound_parts"):
        deeper = _resolve_compound_origin(
            entry.get("compound_parts"),
            entry_path,
            index,
            visited,
            max_depth - 1)
        if deeper:
            return deeper

    entry_lemma = entry.get("lemma") or ""
    if entry_lemma and not _is_informative_origin(direct_origin):
        entry_plain = _HOMOGRAPH_RE.sub("", entry_lemma).lower()
        for related in entry.get("related_words") or []:
            related_plain = _HOMOGRAPH_RE.sub("", str(related)).lower()
            if not related_plain or related_plain == entry_plain:
                continue
            # accept if it's a proper substring OR shares any 3-char trigram with the lemma
            is_related = related_plain in entry_plain or (
                len(related_plain) >= 3
                and any(related_plain[i:i+3] in entry_plain for i in range(len(related_plain) - 2))
            )
            if not is_related:
                continue
            deeper = _resolve_base_origin(related, entry_path, index, visited, max_depth - 1)
            if _is_informative_origin(deeper):
                return deeper
    return direct_origin


def _patch_derivation_base_origins(json_root: Path) -> int:
    """Second-pass: add derivation_base_origin / compound_parts_origin to all JSON files."""
    index = _build_lemma_index(json_root)
    patched = 0
    for path in sorted(json_root.rglob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        lemma = entry.get("lemma") or path.stem
        changed = False

        current_path = str(path)

        db = entry.get("derivation_base")
        if db and entry.get("derivation_base_origin") in {None, "", "internal"}:
            origin = _resolve_base_origin(db, current_path, index, set())
            if origin:
                entry.pop("derivation_base", None)
                entry.pop("derivation_base_origin", None)
                entry["derivation_base"] = db
                entry["derivation_base_origin"] = origin
                changed = True

        cp = entry.get("compound_parts")
        if cp:
            origins = dict(entry.get("compound_parts_origins") or {})
            for part in cp:
                if origins.get(part) in {None, "", "internal"}:
                    o = _resolve_base_origin(part, current_path, index, set())
                    if o:
                        origins[part] = o
            if origins != (entry.get("compound_parts_origins") or {}):
                entry.pop("compound_parts", None)
                entry.pop("compound_parts_origins", None)
                entry["compound_parts"] = cp
                entry["compound_parts_origins"] = origins
                changed = True

        for sub in entry.get("subentries") or []:
            db = sub.get("derivation_base")
            if db and sub.get("derivation_base_origin") in {None, "", "internal"}:
                origin = (
                    _resolve_base_origin_from_siblings(db, entry)
                    or _resolve_base_origin(db, current_path, index, set())
                )
                if origin:
                    sub.pop("derivation_base", None)
                    sub.pop("derivation_base_origin", None)
                    sub["derivation_base"] = db
                    sub["derivation_base_origin"] = origin
                    changed = True
            cp = sub.get("compound_parts")
            if cp:
                origins = dict(sub.get("compound_parts_origins") or {})
                for part in cp:
                    if origins.get(part) in {None, "", "internal"}:
                        o = _resolve_base_origin(part, current_path, index, set())
                        if o:
                            origins[part] = o
                if origins != (sub.get("compound_parts_origins") or {}):
                    sub.pop("compound_parts", None)
                    sub.pop("compound_parts_origins", None)
                    sub["compound_parts"] = cp
                    sub["compound_parts_origins"] = origins
                    changed = True

        if changed:
            entry = _reorder_formation_fields(entry)
            if "subentries" in entry:
                entry["subentries"] = [
                    _reorder_formation_fields(s) for s in entry["subentries"]]
            path.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            patched += 1
    return patched


def split_whole_file_into_entry_blocks(raw_text: str) -> list[str]:

    raw_lines = raw_text.splitlines()

    blocks: list[str] = []
    current_block: list[str] = []
    seen_etym_or_book = False

    for i, raw_line in enumerate(raw_lines):
        plain_line = nm.normalize_plain_text(raw_line)

        if not current_block and not plain_line:
            continue

        is_new_start = ep.is_article_start_line_at(raw_lines, i)

        current_nonempty = [ln for ln in current_block
            if nm.normalize_plain_text(ln)]

        current_is_redirect_only = (
            len(current_nonempty) == 1
            and ep.is_redirect_line(nm.normalize_plain_text(current_nonempty[0])))

        should_split = False

        if is_new_start and current_block:
            # Redirects should always stand alone
            if current_is_redirect_only:
                should_split = True

            # Once etymology/book started, a strong new start opens a new article
            elif seen_etym_or_book and ep.is_strong_article_start_after_etymology(raw_lines, i):
                should_split = True

        if should_split:
            block_text = "\n".join(current_block).strip()
            if block_text:
                blocks.append(block_text)

            current_block = []
            seen_etym_or_book = False

        current_block.append(raw_line)

        if ep.is_etymology_start_line(plain_line) or ep.is_book_start_line(plain_line):
            seen_etym_or_book = True

    if current_block:
        block_text = "\n".join(current_block).strip()
        if block_text:
            blocks.append(block_text)

    return blocks

def process_one_txt_file(
    txt_path: Path, output_root: Path,
    start_index: int) -> tuple[int, int]:

    raw_text = txt_path.read_text(encoding="utf-8")

    entry_blocks = split_whole_file_into_entry_blocks(raw_text)
    
    out_subdir = output_root / txt_path.stem
    out_subdir.mkdir(parents=True, exist_ok=True)    
    
    entry_count_saved = 0
    
    current_index = start_index
    
    for block in entry_blocks:

        entry = extract_entry_block(block)        
        
        if entry.get("entry_type") == "redirect": continue
        
        lemma, homograph = nm.get_entry_file_base(entry)
        lemma_safe = nm.normalize_filename(lemma)
        
        if homograph:
            filename = f"{current_index:06d}_{lemma_safe}_{homograph}.json"
        else:
            filename = f"{current_index:06d}_{lemma_safe}.json"

        out_path = out_subdir / filename

        entry = _reorder_formation_fields(entry)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        entry_count_saved += 1
        current_index += 1

    return entry_count_saved, current_index

def process_all_uesz_txt_files(
    input_dir: Path, output_dir: Path,
    pattern: str = "uesz-*.txt") -> None:

    txt_files = sorted(input_dir.glob(pattern)) # only the ones that match the pattern, that is the ones with the entries
    
    if not txt_files:
        print(f"No files found matching: {pattern}")
        return
    
    print(f"Found {len(txt_files)} TXT files.\n")
    
    total_saved = 0
    total_start = time.perf_counter()
    global_index = 1
    
    for txt_path in txt_files:
        print(f"Processing: {txt_path.name}")
        file_start = time.perf_counter()

        number_of_entries_per_letter, global_index = process_one_txt_file(
            txt_path=txt_path,
            output_root=output_dir,
            start_index=global_index)
        
        file_elapsed = time.perf_counter() - file_start
        print(f"  Saved {number_of_entries_per_letter} entries in {file_elapsed:.2f}s.\n")
        total_saved += number_of_entries_per_letter

    total_elapsed = time.perf_counter() - total_start
    print(f"Done. Total saved entries: {total_saved} in {total_elapsed:.2f}s")

    print("Running second pass: resolving derivation_base_origin / compound_parts_origins")
    patch_start = time.perf_counter()
    patched = _patch_derivation_base_origins(output_dir)
    patch_elapsed = time.perf_counter() - patch_start
    print(f"  Patched {patched} entries in {patch_elapsed:.2f}s")
