"""
entry_builder.py
"""

import re
from typing import Any, Optional

from . import regexes as rx
from . import ueszweb_abbreviations as ab
from . import normalization as nm
from . import entry_predicates as ep
from . import block_extraction as bx

# EMPTY STRUCTURE FACTORIES
def make_empty_family_etymology() -> dict[str, Any]:
    return {
        "shared_etym_sum_raw": "",
        "shared_etym_sum_plain": "",
        "shared_blocks": [],
        "other_etym_text_raw": "",
        "other_etym_text_plain": "",
    }


def make_empty_subentry_etymology() -> dict[str, Any]:
    return {
        "etym_sum_raw": "",
        "etym_sum_plain": "",
        "blocks": [],
        "other_text_raw": "",
        "other_text_plain": "",
    }


def make_empty_subentry(index: int, head_line_raw: str = "") -> dict[str, Any]:
    return {
        "subentry_index": index,

        "head_line_raw": head_line_raw,
        "head_line_plain": nm.normalize_plain_text(head_line_raw),

        "lemma": None,
        "homograph": None,
        "pos": None,
        "stemtype": None,

        "flags": {
            "is_obsolete": False,
            "is_archaic": False,
            "is_dialectal": False,
        },

        "corresponding_donor_word": None,
        "first_year_of_attestation": None,
        "origin_type": None,
        "formation_type": None,
        "origin_certainty_level": None,
        "certainty_label": None,
        "primary_donor_language": None,
        "primary_donor_family": None,
        "other_donor_languages": [],
        "related_words": [],
        "origin_hypotheses": [],
        "formation_hypotheses": [],
        "cross_referenced_bases": [],

        "attestations_raw": "",
        "attestations_plain": "",
        "attestations_parsed": [],

        "meanings_raw": "",
        "meanings_plain": "",
        "meanings_parsed": [],

        "derivations_raw": "",
        "derivations_plain": "",
        "derivations_parsed": [],

        "subentry_etymology": make_empty_subentry_etymology(),

        "references_raw": [],
        "references_plain": [],

        "fossilized_inflected_forms_raw": "",
        "fossilized_inflected_forms_plain": "",
        "fossilized_inflected_forms_parsed": [],
    }

#  FAMILY ENTRY / SINGLE ENTRY
def make_word_family_entry(raw_text: str, head_lines_raw: list[str]) -> dict[str, Any]:
    subentries = [
        make_empty_subentry(index=i, head_line_raw=head)
        for i, head in enumerate(head_lines_raw, start=1)
    ]

    return {
        "entry_type": "family",
        "raw_text": raw_text,
        "plain_text": nm.normalize_plain_text(raw_text),

        "number_of_entries": len(subentries),
        "base_of_word_family": [],

        "head_lines_raw": head_lines_raw,
        "head_lines_plain": [nm.normalize_plain_text(h) for h in head_lines_raw],

        "family_etymology": make_empty_family_etymology(),

        "subentries": subentries,

        "book_block_raw": [],
        "book_block_plain": [],

        "see_also_raw": [],
        "see_also_plain": [],

        "unparsed_raw": [],
        "unparsed_plain": [],

        "parser_notes": [],
        "parser_warnings": [],
    }

# REDIRECT ENTRY
def make_redirect_entry(raw_text: str, line: str) -> dict[str, Any]:
    plain = nm.normalize_plain_text(line)
    m = rx.REFERENCE_WITH_ARROW_ENTRY.match(plain)

    if not m:
        return {
            "entry_type": "redirect",
            "raw_text": raw_text,
            "plain_text": nm.normalize_plain_text(raw_text),
            "parser_warnings": ["Redirect entry did not match redirect pattern cleanly."],
        }

    from_lemma = m.group(1)
    from_sup = m.group(2)
    to_lemma = m.group(3)
    to_sup = m.group(4)

    from_hom = int(from_sup.translate(rx.SUPERSCRIPT_MAP)) if from_sup else None
    to_hom = int(to_sup.translate(rx.SUPERSCRIPT_MAP)) if to_sup else None

    return {
        "entry_type": "redirect",
        "raw_text": raw_text,
        "plain_text": nm.normalize_plain_text(raw_text),
        "lemma": from_lemma,
        "homograph": from_hom,
        "reference_to": {
            "lemma": to_lemma,
            "homograph": to_hom,
        },
        "book_block_raw": [],
        "book_block_plain": [],
        "parser_notes": [],
        "parser_warnings": [],
    }

# SUFFIX FAMILY ENTRY
def make_suffix_family_entry(
    raw_entry_text: str,
    lines: list[dict[str, str]],
) -> dict[str, Any]:
    head_line = None
    for ln in lines:
        if rx.SUFFIX_FAMILY_HEAD_RE.match(ln["plain"].strip()):
            head_line = ln
            break

    if not head_line:
        return {
            "entry_type": "unknown",
            "raw_text": raw_entry_text,
            "plain_text": nm.normalize_plain_text(raw_entry_text),
            "parser_warnings": ["Suffix-family entry detected, but no suffix-family head found."],
        }

    m = rx.SUFFIX_FAMILY_HEAD_RE.match(head_line["plain"].strip())
    suffix_base = m.group(1) if m else None
    hom_sup = m.group(2) if m else None
    homograph = int(hom_sup.translate(rx.SUPERSCRIPT_MAP)) if hom_sup else None
    lemma = f"-{suffix_base}" if suffix_base else None

    entry_obj = {
        "entry_type": "suffix_family",
        "raw_text": raw_entry_text,
        "plain_text": nm.normalize_plain_text(raw_entry_text),

        "lemma": lemma,
        "homograph": homograph,
        "display_head": lemma,

        "number_of_entries": 1,
        "base_of_word_family": [0],

        "head_lines_raw": [head_line["raw"]],
        "head_lines_plain": [head_line["plain"]],

        "redirects": [],
        "roman_groups": [],
        "member_forms": [],

        "family_etymology": make_empty_family_etymology(),
        "subentries": [],

        "book_block_raw": [],
        "book_block_plain": [],

        "see_also_raw": [],
        "see_also_plain": [],

        "unparsed_raw": [],
        "unparsed_plain": [],

        "parser_notes": [],
        "parser_warnings": [],
    }

    # Collect lines after head before etymology
    body_lines = []
    head_found = False
    for ln in lines:
        plain = ln["plain"].strip()
        if not head_found:
            if ln is head_line:
                head_found = True
            continue
        if ep.is_etymology_start_line(plain) or ep.is_book_start_line(plain):
            break
        body_lines.append(ln)

    roman_groups = bx.split_lines_into_roman_groups(body_lines)
    all_members = []

    for group in roman_groups:
        roman = group["roman"]
        raw_text_g = " ".join(ln["raw"].strip() for ln in group["lines"] if ln["raw"].strip())
        plain_text_g = " ".join(ln["plain"].strip() for ln in group["lines"] if ln["plain"].strip())

        raw_text_g = re.sub(r"\s+", " ", raw_text_g).strip()
        plain_text_g = re.sub(r"\s+", " ", plain_text_g).strip()
        plain_text_g = re.sub(
            r'^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*', '', plain_text_g, count=1
        )

        members = []

        for w in re.findall(rf'\b([{rx.HUN_LETTERS}]+féle)\b', plain_text_g):
            members.append(w)

        for w in re.findall(rf'\b([{rx.HUN_LETTERS}]+~)\b', plain_text_g):
            members.append(w)

        members = list(dict.fromkeys(members))
        all_members.extend(members)

        entry_obj["roman_groups"].append({
            "roman": roman,
            "members": members,
        })

    entry_obj["member_forms"] = list(dict.fromkeys(all_members))

    ety_chunks = bx.extract_etymology_chunks(lines)
    if ety_chunks:
        first_chunk = ety_chunks[0]
        entry_obj["family_etymology"]["shared_etym_sum_raw"] = first_chunk["raw_text"]
        entry_obj["family_etymology"]["shared_etym_sum_plain"] = first_chunk["plain_text"]
        entry_obj["family_etymology"]["shared_blocks"] = bx.extract_symbolic_etymology_blocks(
            first_chunk["raw_text"]
        )

    book_raw, book_plain = bx.extract_book_block(lines)
    entry_obj["book_block_raw"] = [" ".join(x.strip() for x in book_raw if x.strip())] if book_raw else []
    entry_obj["book_block_plain"] = [" ".join(x.strip() for x in book_plain if x.strip())] if book_plain else []

    return entry_obj

# PREFIX FAMILY ENTRY
def make_prefix_family_entry(
    raw_entry_text: str,
    lines: list[dict[str, str]],
) -> dict[str, Any]:
    prefix_head_line = bx.get_prefix_head_line(lines)

    if not prefix_head_line:
        return {
            "entry_type": "unknown",
            "raw_text": raw_entry_text,
            "plain_text": nm.normalize_plain_text(raw_entry_text),
            "parser_warnings": ["Prefix-family entry detected, but no prefix head found."],
        }

    m = rx.PREFIX_FAMILY_HEAD_RE.match(prefix_head_line["plain"].strip())
    base = m.group(1) if m else None
    hom_sup = m.group(2) if m else None
    homograph = int(hom_sup.translate(rx.SUPERSCRIPT_MAP)) if hom_sup else None
    display_head = f"{base}-" if base else None
    lemma = display_head or base

    entry_obj = {
        "entry_type": "prefix_family",
        "raw_text": raw_entry_text,
        "plain_text": nm.normalize_plain_text(raw_entry_text),

        "lemma": lemma,
        "homograph": homograph,
        "display_head": display_head,

        "number_of_entries": 1,
        "base_of_word_family": [0],

        "head_lines_raw": [prefix_head_line["raw"]],
        "head_lines_plain": [prefix_head_line["plain"]],

        "redirects": bx.collect_leading_redirects_before_prefix(lines),

        "family_etymology": make_empty_family_etymology(),

        "subentries": [],
        "tilde_members": [],

        "book_block_raw": [],
        "book_block_plain": [],

        "see_also_raw": [],
        "see_also_plain": [],

        "unparsed_raw": [],
        "unparsed_plain": [],

        "parser_notes": [],
        "parser_warnings": [],
    }

    body_lines = bx.get_lines_after_prefix_before_etymology(lines)

    for ln in body_lines:
        plain = ln["plain"].strip()
        for m2 in re.finditer(rf'~([{rx.HUN_LETTERS}]+(?:-[{rx.HUN_LETTERS}]+)*)', plain):
            entry_obj["tilde_members"].append("~" + m2.group(1))

    entry_obj["tilde_members"] = list(dict.fromkeys(entry_obj["tilde_members"]))

    ety_chunks = bx.extract_etymology_chunks(lines)
    if ety_chunks:
        first_chunk = ety_chunks[0]
        entry_obj["family_etymology"]["shared_etym_sum_raw"] = first_chunk["raw_text"]
        entry_obj["family_etymology"]["shared_etym_sum_plain"] = first_chunk["plain_text"]
        entry_obj["family_etymology"]["shared_blocks"] = bx.extract_symbolic_etymology_blocks(
            first_chunk["raw_text"]
        )

        if len(ety_chunks) > 1:
            for extra in ety_chunks[1:]:
                entry_obj["family_etymology"]["other_etym_text_raw"] += (
                    ("\n" if entry_obj["family_etymology"]["other_etym_text_raw"] else "")
                    + extra["raw_text"]
                )
                entry_obj["family_etymology"]["other_etym_text_plain"] += (
                    ("\n" if entry_obj["family_etymology"]["other_etym_text_plain"] else "")
                    + extra["plain_text"]
                )

    book_raw, book_plain = bx.extract_book_block(lines)
    entry_obj["book_block_raw"] = [" ".join(x.strip() for x in book_raw if x.strip())] if book_raw else []
    entry_obj["book_block_plain"] = [" ".join(x.strip() for x in book_plain if x.strip())] if book_plain else []

    return entry_obj

# ROMAN-GROUPED TILDE FAMILY ENTRY
def make_roman_grouped_tilde_entry(
    raw_entry_text: str,
    lines: list[dict[str, str]],
) -> dict[str, Any]:
    prefix_head_line = bx.get_prefix_head_line(lines)

    if not prefix_head_line:
        return {
            "entry_type": "unknown",
            "raw_text": raw_entry_text,
            "plain_text": nm.normalize_plain_text(raw_entry_text),
            "parser_warnings": ["Roman-grouped tilde entry detected, but no prefix head found."],
        }

    m = rx.PREFIX_FAMILY_HEAD_RE.match(prefix_head_line["plain"].strip())
    base = m.group(1) if m else None
    display_head = f"{base}-" if base else None
    lemma = display_head or base

    entry_obj = {
        "entry_type": "roman_grouped_tilde_family",
        "raw_text": raw_entry_text,
        "plain_text": nm.normalize_plain_text(raw_entry_text),

        "lemma": lemma,
        "display_head": display_head,

        "number_of_entries": 1,
        "base_of_word_family": [0],

        "head_lines_raw": [prefix_head_line["raw"]],
        "head_lines_plain": [prefix_head_line["plain"]],

        "redirects": bx.collect_leading_redirects_before_prefix(lines),
        "roman_groups": [],
        "tilde_members": [],

        "family_etymology": make_empty_family_etymology(),
        "subentries": [],

        "book_block_raw": [],
        "book_block_plain": [],

        "see_also_raw": [],
        "see_also_plain": [],

        "unparsed_raw": [],
        "unparsed_plain": [],

        "parser_notes": [],
        "parser_warnings": [],
    }

    # Roman groups and tilde-member collection
    body_lines = bx.get_lines_after_prefix_before_etymology(lines)
    roman_groups = bx.split_lines_into_roman_groups(body_lines)

    roman_group_meta = []
    all_tilde_members = []

    for group in roman_groups:
        roman = group["roman"]
        members = bx.split_roman_group_into_tilde_members(group["lines"])

        member_lemmas = []
        for member in members:
            m2 = rx.TILDE_MEMBER_START_RE.match(member["plain"])
            if m2:
                tilde_form = "~" + m2.group(1)
                member_lemmas.append(tilde_form)
                all_tilde_members.append(tilde_form)

        roman_group_meta.append({
            "roman": roman,
            "members": member_lemmas,
        })

    entry_obj["roman_groups"] = roman_group_meta
    entry_obj["tilde_members"] = list(dict.fromkeys(all_tilde_members))

    # Shared family etymology
    ety_chunks = bx.extract_etymology_chunks(lines)
    if ety_chunks:
        first_chunk = ety_chunks[0]
        entry_obj["family_etymology"]["shared_etym_sum_raw"] = first_chunk["raw_text"]
        entry_obj["family_etymology"]["shared_etym_sum_plain"] = first_chunk["plain_text"]
        entry_obj["family_etymology"]["shared_blocks"] = bx.extract_symbolic_etymology_blocks(
            first_chunk["raw_text"]
        )

        if len(ety_chunks) > 1:
            for extra in ety_chunks[1:]:
                entry_obj["family_etymology"]["other_etym_text_raw"] += (
                    ("\n" if entry_obj["family_etymology"]["other_etym_text_raw"] else "")
                    + extra["raw_text"]
                )
                entry_obj["family_etymology"]["other_etym_text_plain"] += (
                    ("\n" if entry_obj["family_etymology"]["other_etym_text_plain"] else "")
                    + extra["plain_text"]
                )

    # Bibliography
    book_raw, book_plain = bx.extract_book_block(lines)
    entry_obj["book_block_raw"] = [" ".join(x.strip() for x in book_raw if x.strip())] if book_raw else []
    entry_obj["book_block_plain"] = [" ".join(x.strip() for x in book_plain if x.strip())] if book_plain else []

    return entry_obj
