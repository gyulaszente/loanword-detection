"""
Predicates for recognizing ÚESz entry boundaries and entry shapes.
These helpers return only booleans.
"""

from . import regexes as rx
from . import ueszweb_abbreviations as ab
from . import normalization as nm
import re

def starts_new_entry(plain_line: str) -> bool:
    """Return True when a normalized line can begin any supported entry type."""
    return (
        is_head_line(plain_line)
        or is_redirect_line(plain_line)
        or rx.PREFIX_FAMILY_HEAD_RE.match(plain_line) is not None
        or rx.SUFFIX_FAMILY_HEAD_RE.match(plain_line) is not None)

def is_raw_bold_italic_head_line(raw_line: str) -> bool:
    """Check the HTML-marked source line for a bold-italic headword."""
    if not raw_line:
        return False
    return rx.RAW_BOLD_ITALIC_HEAD_RE.match(raw_line) is not None

def is_flag_only_line(plain_line: str) -> bool:
    return plain_line.strip() in {"×", "†", "△", "∆"}

def is_head_line(plain_line: str) -> bool:
    """Check whether a cleaned line begins with a dictionary headword."""
    return rx.HEAD_RE.match(plain_line) is not None


def is_redirect_line(plain_line: str) -> bool:
    """Detect simple cross-reference entries of the form 'x -> y'."""
    return rx.REFERENCE_WITH_ARROW_ENTRY.match(plain_line) is not None

def is_family_etymology_start_line(plain_line: str) -> bool:
    """Detect the start of an etymology paragraph for a word-family entry."""
    return rx.FAMILY_ETY_START_RE.match(plain_line) is not None

def is_book_start_line(plain_line: str) -> bool:
    """Detect the bibliography block marker."""
    return rx.BOOK_START_RE.match(plain_line) is not None

def is_etymology_start_line(plain_line: str) -> bool:
    """Detect a real filled-square etymology start, ignoring empty square markers."""
    if not plain_line:
        return False

    s = plain_line.lstrip()

    #  ignore empty square
    if s.startswith("□"):
        return False

    # only accept real etymology marker
    return re.match(r'^■\s+\S', s) is not None

def is_multiline_entry_start(raw_lines: list[str], i: int) -> bool:
    """
    Allows small line shifts between head and A:.
    """
    if i >= len(raw_lines):
        return False

    raw0 = raw_lines[i]
    plain0 = nm.normalize_plain_text(raw0)

    # must begin with a true bold+italic head
    if not is_raw_bold_italic_head_line(raw0):
        return False

    # if A: is already on same line, great
    if "A:" in plain0:
        return True

    # otherwise allow a short lookahead window
    for j in range(i + 1, min(i + 5, len(raw_lines))):
        rawj = raw_lines[j]
        plainj = nm.normalize_plain_text(rawj)

        if not plainj:
            continue

        # flag line between head and A:
        if is_flag_only_line(plainj):
            continue

        # A: may appear alone on a line
        if plainj.startswith("A:") or rx.RAW_A_LINE_RE.match(rawj):
            return True

        # anything else means this was not a normal head+A pattern
        break

    return False

def is_article_start_line_at(raw_lines: list[str], i: int) -> bool:
    """
    Decide if the line at index i is a (potential) article-start line.
    """
    raw_line = raw_lines[i]                       # original line with <b>, <i>, etc.
    plain_line = nm.normalize_plain_text(raw_line)   # normalized plain text

    # Empty line cannot start an article
    if not plain_line.strip():
        return False

    # real bold+italic head (possibly multi-line with shifted A:)
    if is_multiline_entry_start(raw_lines, i):
        return True

    # Redirect line like "x → y"
    if rx.REFERENCE_WITH_ARROW_ENTRY.match(plain_line):
        return True

    # Suffix-family head: must match the pattern and be bold+italic in raw
    if rx.SUFFIX_FAMILY_HEAD_RE.match(plain_line) and is_raw_bold_italic_head_line(raw_line):
        return True

    # Prefix-family head: must match the pattern and be bold+italic in raw
    if rx.PREFIX_FAMILY_HEAD_RE.match(plain_line) and is_raw_bold_italic_head_line(raw_line):
        return True

    # Otherwise, not a start
    return False

def is_prefix_family_entry(lines: list[dict[str, str]]) -> bool:

    if not lines:
        return False

    prefix_idx = None
    for i, ln in enumerate(lines[:10]):
        if rx.PREFIX_FAMILY_HEAD_RE.match(ln["plain"].strip()):
            prefix_idx = i
            break

    if prefix_idx is None:
        return False

    for ln in lines[prefix_idx + 1:]:
        plain = ln["plain"].lstrip()

        if plain.startswith("~"):
            return True

        if plain.startswith(("■", "🕮")):
            break

    return False

def is_suffix_family_entry(lines: list[dict[str, str]]) -> bool:
    if not lines:
        return False

    head_idx = None
    for i, ln in enumerate(lines[:10]):
        if rx.SUFFIX_FAMILY_HEAD_RE.match(ln["plain"].strip()):
            head_idx = i
            break

    if head_idx is None:
        return False

    for ln in lines[head_idx + 1:]:
        plain = ln["plain"].strip()

        if rx.ROMAN_NUMERAL_RE.match(plain):
            return True

        if plain.startswith(("■", "🕮")):
            break

    return False

def is_roman_grouped_tilde_entry(lines: list[dict[str, str]]) -> bool:

    if not lines:
        return False

    # must contain a prefix head somewhere near the start
    prefix_idx = None
    for i, ln in enumerate(lines[:8]):
        if rx.PREFIX_FAMILY_HEAD_RE.match(ln["plain"].strip()):
            prefix_idx = i
            break

    if prefix_idx is None:
        return False

    # after that, look for Roman numeral grouped ~members
    for ln in lines[prefix_idx + 1 : prefix_idx + 10]:
        plain = ln["plain"].strip()
        if rx.ROMAN_GROUPED_TILDE_START_RE.match(plain):
            return True

    return False

def is_strong_article_start_after_etymology(raw_lines: list[str], i: int) -> bool:
    raw_line = raw_lines[i]
    plain_line = nm.normalize_plain_text(raw_line)

    if not plain_line.strip():
        return False

    if is_multiline_entry_start(raw_lines, i):
        return True

    if rx.REFERENCE_WITH_ARROW_ENTRY.match(plain_line):
        return True

    # suffix-family after etymology/book only if typographically a real head
    if rx.SUFFIX_FAMILY_HEAD_RE.match(plain_line) and is_raw_bold_italic_head_line(raw_line):
        return True

    # prefix-family after etymology/book only if typographically a real head
    if rx.PREFIX_FAMILY_HEAD_RE.match(plain_line) and is_raw_bold_italic_head_line(raw_line):
        return True

    return False
