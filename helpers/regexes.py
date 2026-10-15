import re

TAG_RE = re.compile(r"</?[bi]>")

SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
SUPERSCRIPT_MAP = str.maketrans(SUPERSCRIPTS, "0123456789")

HUN_LETTERS = "A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű"
LEMMA_CORE = rf"[{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*-?"
HEADWORD_CORE = rf"-?[{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*-?"

ROMAN_NUMERAL_RE = re.compile(
    r'^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*'
)

ROMAN_GROUPED_TILDE_START_RE = re.compile(
    rf'^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*~[{HUN_LETTERS}]'
)

TILDE_MEMBER_START_RE = re.compile(
    rf'^\s*~([{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*)'
)

HEADLIKE_MEMBER_BEFORE_A_RE = re.compile(
    rf'(?:^|\|\s*)'
    rf'('
    rf'~[{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*'          # ~alma
    rf'|'
    rf'[{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*~'          # kutya~
    rf'|'
    rf'[{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*'           # affajta
    rf')'
    rf'\s+A:'
)

ILYENEK_MEG_MEMBER_RE = re.compile(
    rf'Ilyenek még:\s*'
    rf'('
    rf'~[{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*'
    rf'|'
    rf'[{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*~'
    rf'|'
    rf'[{HUN_LETTERS}]+(?:-[{HUN_LETTERS}]+)*'
    rf')'
)


HEAD_RE = re.compile(
    rf'^\s*'
    rf'(?!🕮)(?!~)'
    rf'({HEADWORD_CORE})'
    rf'([{SUPERSCRIPTS}]+)?'
    rf'(?:\s*‹[^›]+›)?'
    rf'\s*[†△×∆]?'
    rf'\s*(\[\d+(?:/\d+)*\])?'
    rf'\s*A:'
)

RAW_BOLD_ITALIC_HEAD_RE = re.compile(
    rf'^\s*<b>\s*<i>\s*({HEADWORD_CORE})\s*</i>\s*</b>'
    rf'(?:\s*<b>\s*([{SUPERSCRIPTS}]+)\s*</b>)?'
)

RAW_A_LINE_RE = re.compile(r'^\s*<b>\s*A:\s*</b>\s*$')

REFERENCE_WITH_ARROW_ENTRY = re.compile(
    rf'^\s*'
    rf'(?![→⇒(])'
    rf'({LEMMA_CORE})'
    rf'([{SUPERSCRIPTS}]+)?'
    rf'\s*[→⇒]\s*'
    rf'({HEADWORD_CORE})'
    rf'([{SUPERSCRIPTS}]+)?'
    rf'\s*$'
)

PREFIX_FAMILY_HEAD_RE = re.compile(
    rf'^\s*'
    rf'([{HUN_LETTERS}]+)-'
    rf'([{SUPERSCRIPTS}]+)?'
    rf'\s*$'
)

SUFFIX_FAMILY_HEAD_RE = re.compile(
    rf'^\s*'
    rf'-([{HUN_LETTERS}]+)'
    rf'([{SUPERSCRIPTS}]+)?'
    rf'\s*$'
)

ANY_AFFIX_FAMILY_HEAD_RE = re.compile(
    rf'^\s*'
    rf'(-?[{HUN_LETTERS}]+-?|-[{HUN_LETTERS}]+)'
    rf'([{SUPERSCRIPTS}]+)?'
    rf'\s*$'
)

BOOK_START_RE = re.compile(r'^\s*🕮')
ETY_START_RE = re.compile(r'^\s*■\s+\S')
FAMILY_ETY_START_RE = re.compile(r'^\s*▣\s+\S')

SECTION_MARKER_RE = re.compile(r'\b(A:|J:|Sz:|R:)\b')

