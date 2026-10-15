"""
block_extraction.py

"""

import re
from typing import Any

from . import regexes as rx
from . import normalization as nm
from . import entry_predicates as ep
from . import ueszweb_abbreviations as ab


def get_prefix_head_line(lines: list[dict[str, str]]) -> dict[str, str] | None:
    """Find the first line matching a prefix-family head like 'deci-'."""
    for ln in lines:
        if rx.PREFIX_FAMILY_HEAD_RE.match(ln["plain"].strip()):
            return ln
    return None


def collect_leading_redirects_before_prefix(
    lines: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """
    Collect redirect entries that appear before the prefix head.
    """
    from . import entry_builder as eb

    redirects = []
    for ln in lines:
        plain = ln["plain"].strip()

        if rx.PREFIX_FAMILY_HEAD_RE.match(plain):
            break

        if ep.is_redirect_line(plain):
            redirects.append(eb.make_redirect_entry(ln["raw"], ln["raw"]))

    return redirects


def get_lines_after_prefix_before_etymology(
    lines: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Return body lines between the prefix head and the first ■ or 🕮."""
    prefix_found = False
    out = []

    for ln in lines:
        plain = ln["plain"].strip()

        if not prefix_found:
            if rx.PREFIX_FAMILY_HEAD_RE.match(plain):
                prefix_found = True
            continue

        if ep.is_etymology_start_line(plain) or ep.is_book_start_line(plain):
            break

        out.append(ln)

    return out

def split_lines_into_roman_groups(
    lines_after_prefix: list[dict[str, str]]
) -> list[dict[str, Any]]:

    groups = []
    current_roman = None
    current_lines = []

    for ln in lines_after_prefix:
        plain = ln["plain"].strip()
        m = rx.ROMAN_NUMERAL_RE.match(plain)

        if m:
            if current_roman is not None:
                groups.append({
                    "roman": current_roman,
                    "lines": current_lines,
                })
            current_roman = m.group(1)
            current_lines = [ln]
        else:
            if current_roman is not None:
                current_lines.append(ln)

    if current_roman is not None:
        groups.append({
            "roman": current_roman,
            "lines": current_lines,
        })

    return groups


def split_roman_group_into_tilde_members(
    group_lines: list[dict[str, str]]
) -> list[dict[str, str]]:

    if not group_lines:
        return []

    raw_text = " ".join(
        ln["raw"].strip() for ln in group_lines if ln["raw"].strip()
    )
    plain_text = " ".join(
        ln["plain"].strip() for ln in group_lines if ln["plain"].strip()
    )

    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    plain_text = re.sub(r"\s+", " ", plain_text).strip()

    raw_text = re.sub(
        r'^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*',
        '', raw_text, count=1
    )
    plain_text = re.sub(
        r'^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*',
        '', plain_text, count=1
    )

    split_pat = rf'(?=(?:\|\s*)?~[{rx.HUN_LETTERS}])'

    raw_parts = [p.strip(" |") for p in re.split(split_pat, raw_text) if p.strip(" |")]
    plain_parts = [p.strip(" |") for p in re.split(split_pat, plain_text) if p.strip(" |")]

    members = []
    max_len = max(len(raw_parts), len(plain_parts))

    for i in range(max_len):
        raw_part = raw_parts[i] if i < len(raw_parts) else ""
        plain_part = plain_parts[i] if i < len(plain_parts) else ""

        if not plain_part.startswith("~"):
            continue

        members.append({
            "raw": raw_part,
            "plain": plain_part,
        })

    return members

def extract_etymology_chunks(
    lines: list[dict[str, str]]
) -> list[dict[str, Any]]:

    chunks = []
    i = 0

    while i < len(lines):
        plain = lines[i]["plain"]

        if not (
            ep.is_etymology_start_line(plain)
            or ep.is_family_etymology_start_line(plain)
        ):
            i += 1
            continue

        intro_symbol = plain.lstrip()[0]
        raw_lines = [lines[i]["raw"]]
        plain_lines = [lines[i]["plain"]]
        i += 1

        while i < len(lines):
            next_plain = lines[i]["plain"]
            if (
                ep.is_etymology_start_line(next_plain)
                or ep.is_family_etymology_start_line(next_plain)
                or ep.is_book_start_line(next_plain)
            ):
                break
            raw_lines.append(lines[i]["raw"])
            plain_lines.append(lines[i]["plain"])
            i += 1

        chunks.append({
            "intro_symbol": intro_symbol,
            "raw_text": "\n".join(raw_lines),
            "plain_text": "\n".join(plain_lines),
        })

    return chunks


def extract_symbolic_etymology_blocks(
    chunk_raw_text: str,
) -> list[dict[str, Any]]:

    chunk_plain_text = nm.normalize_plain_text(chunk_raw_text)

    symbol_list = sorted(ab.ETY_BLOCK_ROLE_MAP.keys(), key=len, reverse=True)
    symbol_pattern = "(" + "|".join(re.escape(sym) for sym in symbol_list) + ")"

    blocks = []
    current_symbol = None
    current_raw_parts = []
    current_plain_parts = []

    chunk_raw_wo_intro = re.sub(r'^\s*[■▣]\s*', '', chunk_raw_text, count=1)
    chunk_plain_wo_intro = re.sub(r'^\s*[■▣]\s*', '', chunk_plain_text, count=1)

    raw_segments = [seg.strip() for seg in chunk_raw_wo_intro.split("|") if seg.strip()]
    plain_segments = [seg.strip() for seg in chunk_plain_wo_intro.split("|") if seg.strip()]

    max_segments = max(len(raw_segments), len(plain_segments), 1)

    for seg_idx in range(max_segments):
        raw_seg = raw_segments[seg_idx] if seg_idx < len(raw_segments) else ""
        plain_seg = plain_segments[seg_idx] if seg_idx < len(plain_segments) else ""

        plain_parts = re.split(symbol_pattern, plain_seg)
        raw_parts = re.split(symbol_pattern, raw_seg)

        max_parts = max(len(plain_parts), len(raw_parts))

        for idx in range(max_parts):
            p = plain_parts[idx].strip() if idx < len(plain_parts) and plain_parts[idx] else ""
            r = raw_parts[idx].strip() if idx < len(raw_parts) and raw_parts[idx] else ""

            token = p if p else r
            if not token:
                continue

            if token in ab.ETY_BLOCK_ROLE_MAP:

                if current_symbol is not None:
                    blocks.append({
                        "symbol": current_symbol,
                        "role": ab.ETY_BLOCK_ROLE_MAP[current_symbol],
                        "description": ab.UESZ_SIGNS_ETYMOLOGY_BLOCKS.get(current_symbol, ""),
                        "text_raw": " ".join(current_raw_parts).strip(),
                        "text_plain": " ".join(current_plain_parts).strip(),
                    })

                current_symbol = token
                current_raw_parts = []
                current_plain_parts = []
            else:
                if current_symbol is not None:
                    if r:
                        current_raw_parts.append(r)
                    if p:
                        current_plain_parts.append(p)

    if current_symbol is not None:
        blocks.append({
            "symbol": current_symbol,
            "role": ab.ETY_BLOCK_ROLE_MAP[current_symbol],
            "description": ab.UESZ_SIGNS_ETYMOLOGY_BLOCKS.get(current_symbol, ""),
            "text_raw": " ".join(current_raw_parts).strip(),
            "text_plain": " ".join(current_plain_parts).strip(),
        })

    return blocks

def extract_book_block(
    lines: list[dict[str, str]]
) -> tuple[list[str], list[str]]:

    book_raw = []
    book_plain = []

    collecting = False
    for line in lines:
        plain = line["plain"]

        if ep.is_book_start_line(plain):
            collecting = True
            book_raw.append(line["raw"])
            book_plain.append(line["plain"])
            continue

        if collecting:
            if ep.starts_new_entry(plain) or ep.is_etymology_start_line(plain):
                break
            book_raw.append(line["raw"])
            book_plain.append(line["plain"])

    return book_raw, book_plain
