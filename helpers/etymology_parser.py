"""
etymology_parser.py
Main parsing logic for ÚESz dictionary entries.
"""

import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any
from . import regexes as rx
from . import normalization as nm
from . import entry_predicates as ep
from . import entry_builder as eb
from . import block_extraction as bx
from . import head_extraction as he
from . import summary_normalization as sn
from . import ueszweb_abbreviations as ab

# ---------------------------------------------------------------

def _repair_mojibake(text: str | None) -> str | None:
    """ repairs broken characters, '?' and weird-looking squares etc. """
    if text is None:
        return None
    if "Ã" not in text and "Å" not in text:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text



""" ETYMOLOGY BLOCK SYMBOLS ordered by length so multi-char ones match first
Two-character symbols MUST come before their single-char components """
ETY_SYMBOLS_ORDERED = ["⇒⌂", "⌂⇒", "≡", "⌂", "≋", "∼", "≁", "≂", "⚠", "⌘", "⊚"]


# ------------- REGEXES ---------------------------------------------------------
ETY_SYMBOL_SPLIT_RE = re.compile(
    r'(?=\s*(?:' + '|'.join(re.escape(s) for s in ETY_SYMBOLS_ORDERED) + r')\s)')

LANG_ALIAS_TO_NAME = {}
for _lang, _forms in ab.ABBREVIATIONS_DICT.items():
    LANG_ALIAS_TO_NAME[_lang.casefold()] = _lang
    for _form in _forms:
        LANG_ALIAS_TO_NAME[_form.casefold()] = _lang

LANG_HU_TO_EN = dict(getattr(ab, "LANGUAGE_NAME_ENGLISH", {}))
for _hu, _en in list(LANG_HU_TO_EN.items()):
    _repaired = _repair_mojibake(_hu)
    if _repaired and _repaired not in LANG_HU_TO_EN:
        LANG_HU_TO_EN[_repaired] = _en
LANG_EN_TO_HU = {en.casefold(): hu for hu, en in LANG_HU_TO_EN.items()}

LANG_ALIAS_ITEMS = [
    (alias, lang, re.compile(rf'(?<!\w){re.escape(alias)}(?!\w)'))
    for alias, lang in sorted(LANG_ALIAS_TO_NAME.items(), key=lambda item: len(item[0]), reverse=True)
    if len(alias) >= 3 or "." in alias]

LANG_NAME_TO_FAMILY = {
    lang.casefold(): family
    for family, langs in ab.LANGUAGE_FAMILIES.items()
    for lang in langs}
LANG_NAME_TO_FAMILY.update({
    _repair_mojibake(lang).casefold(): family
    for family, langs in ab.LANGUAGE_FAMILIES.items()
    for lang in langs
    if _repair_mojibake(lang)})
# ------------- REGEXES ---------------------------------------------------------




#-------------------------------------------------------------------------------
""" normalization mapping """

ETY_SUMMARY_TAG_TO_ORIGIN = {
    "<INHERITED>": "inherited",
    "<CALQUE>": "calque",
    "<LOANBLEND>": "loanblend",
    "<LOAN_WORD>": "loan",
    "<INTERNATIONAL>": "international",
    "<WANDERWORD>": "wanderword",
    "<ONOMATOPOETIC>": "onomatopoetic",
    "<FICTIONAL_PLAYFUL>": "fictional",
    "<INTERNAL_ORIGIN>": "internal",
    "<UNKNOWN_ORIGIN>": "unknown",
}

ETY_SUMMARY_TAG_TO_FORMATION = {
    "<DERIVATION>": "derivation",
    "<COMPOUND>": "compound",
    "<PHRASAL_COMPOUND>": "phrasal_compound",
    "<CONVERSION>": "conversion",
    "<BACK_FORMATION>": "back_formation",
    "<REDUPLICATION>": "reduplication",
    "<FOSSILIZATION>": "fossilization",
    "<CONSTITUENT_LEXICALIZATION>": "constituent_lexicalization",
    "<LEXICAL_SPLIT>": "lexical_split",
    "<COINAGE>": "coinage",
    "<NYELVUJITAS>": "language_reform",
}

POSSIBLE_DONOR_BLOCK_ROLES = {
    "donor_data",
    "in_hungarian",
    "into_hungarian",
}

""" regexes """
POSSIBLE_LOAN_RE = re.compile(
    r'\bj[öo]vev[ée]nysz[óo]\w*'
    r'(?:\W+\w+){0,6}?\W+'
    r'(?:lehet\w*|t[űu]nik|látszik)',
    re.IGNORECASE,
)
EXPLANATORY_NONNATIVE_RE = re.compile(
    r'(részfordítás|tükörfordítás|átvétel|átvétele)', # --> loanwords with native elements
    re.IGNORECASE,
)


# ---------------- HELPER FUNCTIONS FOR NORMALIZATION ----------------------- #

def _find_language_mentions(text: str) -> list[str]:
    if not text:
        return []

    found = []
    lowered = text.casefold()
    for alias, lang, pattern in LANG_ALIAS_ITEMS:
        if alias not in lowered:
            continue
        for match in pattern.finditer(lowered):
            found.append((match.start(), -len(alias), lang))

    return list(dict.fromkeys(lang for _, _, lang in sorted(found)))


def _language_family(lang: str | None) -> str | None:
    if not lang:
        return None
    repaired = _repair_mojibake(str(lang))
    canonical = LANG_EN_TO_HU.get(repaired.casefold(), repaired)
    return LANG_NAME_TO_FAMILY.get(canonical.casefold())


def _english_language_name(lang: str | None) -> str | None:
    if not lang:
        return None
    repaired = _repair_mojibake(str(lang))
    canonical = LANG_EN_TO_HU.get(repaired.casefold(), repaired)
    return LANG_HU_TO_EN.get(canonical, canonical)


def _english_language_names(langs: list[str] | None) -> list[str]:
    if not langs:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for lang in langs:
        en = _english_language_name(lang)
        if en and en not in seen:
            seen.add(en)
            out.append(en)
    return out


def _first_attestation_year(text: str) -> int | None:
    """Fallback year extractor for unparsed attestation strings"""
    years = []
    for segment in re.split(r"[;|]", text or ""):
        m = re.search(r'^\s*(?:\(?†?\s*)?(?:\[[^\]]+\]\s*)?(\d{3,4})(?!\d)', segment)
        if m:
            years.append(int(m.group(1)))
    return min(years) if years else None


def _first_attestation_item(sub: dict[str, Any]) -> dict[str, Any] | None:
    """Return the earliest usable parsed attestation item"""
    candidates = [
        item
        for item in sub.get("attestations_parsed", [])
        if (isinstance(item, dict)
            and item.get("year") is not None
            and item.get("date_type") != "excluded_antedated")
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: int(item["year"]))


def _set_first_year_from_attestations(sub: dict[str, Any]) -> None:
    """Set first-year fields and preserve whether the selected year was generated"""
    first = _first_attestation_item(sub)
    if first is None:
        sub["first_year_of_attestation"] = _first_attestation_year(sub.get("attestations_plain") or "")
        return

    sub["first_year_of_attestation"] = int(first["year"])
    if first.get("year_is_generated"):
        sub["first_year_of_attestation_is_generated"] = True
        sub["first_year_of_attestation_generated_from"] = first.get("year_generated_from")
        sub["first_year_of_attestation_range"] = first.get("year_range")
        sub["first_year_of_attestation_generation_method"] = first.get("year_generation_method")


def _first_italic_word(text: str) -> str | None:
    for m in re.finditer(r'<i>\s*([^<]+?)\s*</i>', text or ""):
        prefix = (text or "")[max(0, m.start() - 3):m.start()]
        if "→" in prefix:
            continue
        return nm.normalize_plain_text(m.group(1))
    return None


def _ascii_skeleton(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = nm.normalize_plain_text(text).casefold()
    return "".join(ch for ch in text if ch.isalpha())


def _lemma_word_similarity(lemma: str, word: str) -> float:
    """
    Return a rough spelling-similarity score for two different forms.
    """
    a = _ascii_skeleton(lemma)
    b = _ascii_skeleton(word)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 4 and a in b:
        return max(0.9, len(a) / len(b))
    if len(b) >= 4 and b in a:
        return max(0.9, len(b) / len(a))
    return SequenceMatcher(None, a, b).ratio()


def _collect_etymology_blocks(etym: dict[str, Any]) -> list[dict[str, str]]:
    blocks = list(etym.get("blocks") or [])
    blocks.extend(etym.get("shared_blocks") or [])
    for analysis in etym.get("analyses") or []:
        blocks.extend(analysis.get("blocks") or [])
    return blocks


def _looks_like_inherited_summary(text: str) -> bool:
    plain = nm.normalize_plain_text(text or "").casefold()
    if not plain:
        return False
    has_inherited_word = ("örökség" in plain) or ("örökségek" in plain)
    has_uralic_scope = any(token in plain for token in ("uráli", "finnugor", "ugor"))
    return has_inherited_word and has_uralic_scope


def _infer_origin_type(etym: dict[str, Any]) -> str | None:
    summary_text = (
        (etym.get("summary_raw") or "")
        or (etym.get("summary_plain") or "")
        or (etym.get("summary") or ""))
    normalized = etym.get("summary_normalized") or sn.normalize_summary_text(
        summary_text,
        sn.LANG_TO_FAMILY,)
    if "<DISPUTED_ORIGIN>" in normalized:
        return "disputed"

    for tag, origin in ETY_SUMMARY_TAG_TO_ORIGIN.items():
        if tag in normalized:
            return origin

    summary_plain = nm.normalize_plain_text(summary_text).casefold()
    if _looks_like_inherited_summary(summary_text):
        return "inherited"
    if "fiktív tő" in summary_plain or "fiktiv to" in summary_plain:
        return "fictional"
    if "onomatopoetikus" in summary_plain:
        return "onomatopoetic"
    if "ismeretlen eredet" in summary_plain:
        return "unknown"

    for analysis in etym.get("analyses") or []:
        label_norm = sn.normalize_summary_text(analysis.get("label_raw") or "", sn.LANG_TO_FAMILY)
        for tag, origin in ETY_SUMMARY_TAG_TO_ORIGIN.items():
            if tag in label_norm:
                return origin

    return None


def _infer_formation_type(etym: dict[str, Any]) -> str | None:
    normalized = etym.get("summary_normalized") or ""
    for tag, formation in ETY_SUMMARY_TAG_TO_FORMATION.items():
        if tag in normalized:
            return formation

    summary_plain = nm.normalize_plain_text(
        (etym.get("summary_raw") or "")
        or (etym.get("summary_plain") or "")
        or (etym.get("summary") or "")).casefold()
    if _mentions_derivative_word(summary_plain):
        return "derivation"
    if "képzéssel" in summary_plain:
        return "derivation"
    if "tőből származ" in summary_plain:
        return "derivation"
    if "ragszilárdulás" in summary_plain:
        return "fossilization"
    if "szófajváltás" in summary_plain:
        return "conversion"

    for analysis in etym.get("analyses") or []:
        label_norm = sn.normalize_summary_text(analysis.get("label_raw") or "", sn.LANG_TO_FAMILY)
        for tag, formation in ETY_SUMMARY_TAG_TO_FORMATION.items():
            if tag in label_norm:
                return formation

    return None


def _etymology_public_text(etym: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("summary_raw", "summary_plain", "summary"):
        value = etym.get(key)
        if value:
            parts.append(str(value))
    for block_list_key in ("blocks", "shared_blocks"):
        for block in etym.get(block_list_key) or []:
            text = block.get("text_raw") or block.get("text_plain") or block.get("text")
            if text:
                parts.append(str(text))
    return " ".join(parts)


def _mentions_derivative_word(text: str) -> bool:
    plain = nm.normalize_plain_text(text or "").casefold()
    return "származékszó" in plain or "származékszav" in plain



def _mentions_productive_suffix_formation(etym: dict[str, Any]) -> bool:
    parts: list[str] = []
    for key in ("summary_raw", "summary_plain", "summary"):
        if etym.get(key):
            parts.append(str(etym.get(key)))
    for block in _collect_etymology_blocks(etym):
        if block.get("role") == "in_hungarian":
            text = block.get("text_raw") or block.get("text_plain") or block.get("text")
            if text:
                parts.append(str(text))
    text = nm.normalize_plain_text(" ".join(parts)).casefold()
    if not text:
        return False

    explicit_suffix_terms = (
        "igeképző",
        "melléknévképző",
        "igenévképző",
        "névszóképző",
        "főnévképző",
        "gyakorító képző",
        "mozzanatos képző",
        "műveltető képző",
        "visszaható képző",
        "kicsinyítő képző",)
    if any(term in text for term in explicit_suffix_terms):
        return True
    return ("képző" in text
        and any(marker in text for marker in 
                ("végződés", "végződések", "szóvég", "szóvégek")))


""" detect whether the etymology text explicitly says that this lemma was formed 
in Hungarian with a suffix  --> 'keletkezett' OR 'képzö' """
def _lemma_is_named_as_hungarian_derivative(lemma: str | None, etym: dict[str, Any]) -> bool:
    if not lemma:
        return False
    text = nm.normalize_plain_text(_etymology_public_text(etym)).casefold()
    lemma_plain = nm.normalize_plain_text(lemma).casefold()
    if not text or not lemma_plain:
        return False

    for match in re.finditer(rf'(?<!\w){re.escape(lemma_plain)}(?!\w)', text):
        following = text[match.end():match.end() + 50]
        if re.match(r'\s+\w*(?:ból|ből)\b', following):
            continue
        sentence_end = text.find(".", match.end())
        sentence = text[match.start(): sentence_end if sentence_end != -1 else len(text)]
        if "keletkezett" in sentence and "képző" in sentence:
            return True
    return False



def _mentions_folk_etymological_loanblend(etym: dict[str, Any]) -> bool:
    text = nm.normalize_plain_text(_etymology_public_text(etym)).casefold()
    return "népetimológiai" in text or "nepetimologiai" in text


def _resolve_parser_formation_type(*, etym: dict[str, Any], formation_type: str | None,
    cross_referenced_bases: list[str] | None = None,
) -> str | None:
    """
    Avoid marking vague "képzéssel" cases as derivations.
    """
    if formation_type != "derivation":
        return formation_type

    if cross_referenced_bases:
        return formation_type

    etym_text = nm.normalize_plain_text(_etymology_public_text(etym)).casefold()
    if "képzéssel" not in etym_text and "tőből származ" not in etym_text:
        return formation_type
    if _mentions_derivative_word(etym_text):
        return formation_type
    return "simplex"


def _resolve_origin_type_with_family(origin_type: str | None, primary_lang: str | None) -> str | None:

    if origin_type != "loan":
        return origin_type
    if _language_family(primary_lang) == "Uralic":
        return "inherited"
    return origin_type


def _summary_implies_internal_model_based_formation(etym: dict[str, Any]) -> bool:
    summary = nm.normalize_plain_text((etym.get("summary_raw") or "") or 
                                      (etym.get("summary_plain") or "") or 
                                      (etym.get("summary") or "")).casefold()
    return any(phrase in summary for phrase in ("mintára alkotott",
                                                "mintára képzett",
                                                "tudatos szóalkotás",
                                                "szóhasadás",
                                                "elvonás",
                                                "rövidülés",
                                                "szórövidülés",
                                                "nyelvújítási"))


def _non_native_hint_from_blocks(etym: dict[str, Any], origin_type: str | None,
    formation_type: str | None, lemma: str | None = None) -> dict[str, Any] | None:
    """
    find evidencde for loan-ish source in the etymological blocks
    Some entries are classified as internal formations because the immediate
    formation is a lexical split, back-formation, or lexicalized compound part.
    But their explanatory blocks might still say that the source model was a
    calque, partial calque, or foreign borrowing. In those cases, the function
    returns donor metadata so the entry is not treated as source-less native data
    in later tables or experiments.
    """
    if origin_type not in {None, "internal"}:
        return None
    if formation_type not in {"constituent_lexicalization", "back_formation", "lexical_split"}:
        return None

    for block in _collect_etymology_blocks(etym):
        role = block.get("role")
        if role not in {"in_hungarian", "into_hungarian"}: # --> IMPORTANT : "other_languages" was removed because it mentions the word in other languages rather than in Hungarian
            continue

        text_raw = block.get("text_raw") or ""
        text_plain = block.get("text_plain") or nm.normalize_plain_text(text_raw)
        normalized = sn.normalize_summary_text(f"{text_plain} {text_raw}", sn.LANG_TO_FAMILY)
        donor_words = _language_word_pairs_from_block(text_raw) # --> [{"language": "german", "word": "Jäger"}, { ... }, ... ]
        for pair in _language_token_pairs_from_plain_block(text_plain):
            if pair not in donor_words:
                donor_words.append(pair)
                
                
        """ IMPORTANT STEP! IT CHECKS WHETHER THE WORD IS SOMEWHAT SIMILAR TO THE CHECKED WORD 
        0.58 IS ARBITRARY, IT WAS CHOSEN AFTER EXPERIMENTING WITH A FEW VARIATIONS """
        similar_donor_words = [
            pair for pair in donor_words
            if lemma and _lemma_word_similarity(lemma, pair["word"]) >= 0.58] 
        
        
        has_explanatory_signal = (
            "<CALQUE>" in normalized
            or "<LOANBLEND>" in normalized
            or EXPLANATORY_NONNATIVE_RE.search(text_plain)
            or bool(similar_donor_words)) # --> TRue or False
        if not has_explanatory_signal: # --> if it is False then continue
            continue

        langs = list(dict.fromkeys(pair["language"] for pair in donor_words)) # --> languagesfrom the donor words
        if not langs:
            langs = _find_language_mentions(f"{text_plain} {text_raw}") # --> if there is none, find if there are any mentions
        langs = [lang for lang in langs if lang != "magyar"] # --> the language cannot be Hungarian since that is not a donor language
        if not langs:
            continue

        """ fallback steps for the most possible donor words
        1. the most similar word
        2. other donor word
        3. first italic word in the sub-block """
        donor_word = (similar_donor_words[0]["word"]
            if similar_donor_words 
            else donor_words[0]["word"] 
            if donor_words 
            else _first_italic_word(text_raw))
        return {
            "origin_type": "loan",
            "primary_donor_language": _english_language_name(langs[0]),
            "primary_donor_family": _language_family(langs[0]),
            "other_donor_languages": _english_language_names(langs[1:] if donor_words else langs[1:]),
            "corresponding_donor_word": donor_word}

    return None


def _infer_certainty(summary_raw: str, etym: dict[str, Any], origin_type: str | None) -> tuple[int | None, str | None]:
    if origin_type == "unknown":
        return 1, None

    normalized = etym.get("summary_normalized") or ""
    raw = (summary_raw or "").casefold()

    if "<DISPUTED_" in normalized or "<UNCERTAIN_" in normalized or "vitatott" in raw or "bizonytalan" in raw:
        return 1, "uncertain"
    if any(word in raw for word in ("talán", "esetleg", "feltehető", "lehet")):
        return 2, "possible"
    if "valószínű" in raw:
        return 3, "probable"
    if origin_type:
        return 4, "certain"
    return None, None


def _extract_related_words(blocks: list[dict[str, str]]) -> list[str]:
    related = []
    for block in blocks:
        text = block.get("text_plain") or ""
        related.extend(_extract_arrow_references(text))

        if block.get("role") in {"belong_here", "belong_maybe_here", "from_hungarian"}:
            raw = block.get("text_raw") or ""
            related.extend(nm.normalize_plain_text(m) for m in re.findall(r'<i>\s*([^<]+?)\s*</i>', raw))

    return list(dict.fromkeys(w.strip() for w in related if w and w.strip()))


def _language_word_pairs_from_block(text_raw: str) -> list[dict[str, str]]:

    pairs: list[dict[str, str]] = []
    text_raw = text_raw or ""

    for italic_match in re.finditer(r'<i>\s*([^<]+?)\s*</i>', text_raw):
        word = nm.normalize_plain_text(italic_match.group(1))
        if not word:
            continue

        prefix = text_raw[max(0, italic_match.start() - 80):italic_match.start()].casefold()
        best_match: tuple[int, int, str] | None = None
        for alias, lang, pattern in LANG_ALIAS_ITEMS:
            matches = list(pattern.finditer(prefix))
            if not matches:
                continue
            valid_matches = []
            for match in matches:
                before = prefix[:match.start()].rstrip()
                if before.endswith("("):
                    continue
                valid_matches.append(match)
            if not valid_matches:
                continue
            last = valid_matches[-1]
            candidate = (last.start(), len(alias), lang)
            if best_match is None or candidate[:2] > best_match[:2]:
                best_match = candidate

        if best_match is not None:
            pairs.append({"language": best_match[2], "word": word})

    unique_pairs: list[dict[str, str]] = []
    seen = set()
    for pair in pairs:
        key = (pair["language"], pair["word"])
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(pair)
    return unique_pairs


def _language_token_pairs_from_plain_block(text_plain: str) -> list[dict[str, str]]:

    pairs: list[dict[str, str]] = []
    plain = text_plain or ""
    folded = plain.casefold()

    for alias, lang, pattern in LANG_ALIAS_ITEMS:
        for match in pattern.finditer(folded):
            window = plain[match.end():match.end() + 48]
            token_match = re.search(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž-]{4,}", window)
            if not token_match:
                continue
            word = nm.normalize_plain_text(token_match.group(0))
            if not word:
                continue
            pairs.append({"language": lang, "word": word})

    unique_pairs: list[dict[str, str]] = []
    seen = set()
    for pair in pairs:
        key = (pair["language"], pair["word"])
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(pair)
    return unique_pairs


def _secondary_origin_hypotheses_from_blocks(
    etym: dict[str, Any],
    origin_type: str | None,
) -> list[dict[str, Any]]:

    if origin_type not in {None, "unknown", "uncertain"}:
        return []

    hypotheses: list[dict[str, Any]] = []
    for block in _collect_etymology_blocks(etym):
        if block.get("role") not in POSSIBLE_DONOR_BLOCK_ROLES:
            continue

        text_raw = block.get("text_raw") or ""
        text_plain = block.get("text_plain") or nm.normalize_plain_text(text_raw)
        folded = text_plain.casefold()

        if not POSSIBLE_LOAN_RE.search(folded):
            continue

        donor_words = _language_word_pairs_from_block(text_raw)
        langs = list(dict.fromkeys(pair["language"] for pair in donor_words))
        if not langs:
            langs = _find_language_mentions(f"{text_plain} {text_raw}")
        langs = [lang for lang in langs if lang != "magyar"]
        if not langs:
            continue

        donor_word = donor_words[0]["word"] if donor_words else _first_italic_word(text_raw)
        hypothesis: dict[str, Any] = {
            "origin_type": "loan",
            "origin_certainty_level": 2,
            "certainty_label": "possible",
            "primary_donor_language": _english_language_name(langs[0]),
            "primary_donor_family": _language_family(langs[0]),
            "other_donor_languages": _english_language_names(langs[1:] if donor_words else []),
            "corresponding_donor_word": donor_word,
            "source_block_role": block.get("role"),
            "trigger": "possible_loan_in_explanation",
        }
        if donor_words:
            hypothesis["donor_words"] = donor_words
        hypotheses.append(hypothesis)

    return hypotheses


def _analysis_text(analysis: dict[str, Any]) -> str:
    parts = [analysis.get("label_raw") or ""]
    for block in analysis.get("blocks") or []:
        parts.append(block.get("text_raw") or "")
        parts.append(block.get("text_plain") or "")
    return " ".join(parts)


def _infer_origin_from_label(label_raw: str, blocks: list[dict[str, str]]) -> str | None:
    normalized = sn.normalize_summary_text(label_raw or "", sn.LANG_TO_FAMILY)
    label_plain = nm.normalize_plain_text(label_raw or "").casefold()

    for tag, origin in ETY_SUMMARY_TAG_TO_ORIGIN.items():
        if tag in normalized:
            return origin

    if _looks_like_inherited_summary(label_raw):
        return "inherited"

    if "belső keletkezésű" in label_plain:
        return "internal"

    donor_blocks = [b for b in blocks if b.get("role") == "donor_data"]
    if donor_blocks:
        return "loan"

    return None


def _infer_formation_from_label(label_raw: str, origin_type: str | None) -> str | None:
    normalized = sn.normalize_summary_text(label_raw or "", sn.LANG_TO_FAMILY)

    for tag, formation in ETY_SUMMARY_TAG_TO_FORMATION.items():
        if tag in normalized:
            return formation

    label_plain = nm.normalize_plain_text(label_raw or "").casefold()
    if "hangrendi átcsapás" in label_plain:
        return "vowel_harmony_shift"
    if "jelentéselkülönülés" in label_plain:
        return "semantic_split"
    if "képzés" in label_plain or "képzéssel" in label_plain:
        return "derivation"

    if origin_type in {
        "loan",
        "inherited",
        "onomatopoetic",
        "international",
        "wanderword",
        "unknown",
        "uncertain",
    }:
        return "simplex"

    return None


def _hypothesis_origin_type(origin_type: str | None,
    formation_type: str | None, primary_lang: str | None) -> str | None:
    """Return the origin class for a disputed/numbered hypothesis"""
    if origin_type:
        return origin_type

    if primary_lang:
        return "loan"

    if formation_type in {
        "derivation",
        "compound",
        "phrasal_compound",
        "conversion",
        "back_formation",
        "reduplication",
        "fossilization",
        "constituent_lexicalization",
        "lexical_split",
        "vowel_harmony_shift",
        "semantic_split",
        "language_reform",
    }:
        return "internal"

    return None


def _extract_cross_references(text: str) -> list[str]:
    return list(dict.fromkeys(_extract_arrow_references(text)))


def _extract_arrow_references(text: str) -> list[str]:
    """Extract arrow-linked lemmas and strip attached Hungarian suffixes"""
    refs: list[str] = []
    plain = nm.normalize_plain_text(text or "")
    for word in re.findall(r'→\s*([^\s,.;:)]+)', plain):
        cleaned = word.strip()
        cleaned = re.sub(r'-[a-záéíóöőúüű]+$', '', cleaned, flags=re.IGNORECASE)
        if cleaned:
            refs.append(cleaned)
    return refs


_ITALIC_WORD_RE = re.compile(r'<i>\s*([^<]+?)\s*</i>')


def _italic_words_from_blocks(blocks: list[dict], max_count: int = 0) -> list[str]:
    """Return italic words from blocks in order, skipping suffix tokens (starting with -)"""
    results: list[str] = []
    for block in blocks:
        raw = block.get("text_raw") or ""
        for m in _ITALIC_WORD_RE.finditer(raw):
            word = nm.normalize_plain_text(m.group(1))
            if word and not word.startswith("-") and word not in results:
                results.append(word)
                if max_count and len(results) >= max_count:
                    return results
    return results


def _strong_italic_formation_bases(blocks: list[dict]) -> list[str]:
    """Return words used in strong Hungarian base-source phrases"""
    bases: list[str] = []
    source_nouns = ("szóból",
                    "főnévből",
                    "igéből",
                    "melléknévből",
                    "alakból",
                    "változatból")
    source_pattern = rf"\s+(?:{'|'.join(source_nouns)})\b"
    result_pattern = r"\b(?:keletkez\w*|jött létre|származ\w*|elvonás eredménye|eredménye)\b"
    word_chars = r"a-záéíóöőúüűÁÉÍÓÖŐÚÜŰ0-9¹²³⁴⁵⁶⁷⁸⁹⁰"

    for block in blocks:
        raw = block.get("text_raw") or block.get("text") or ""
        for match in _ITALIC_WORD_RE.finditer(raw):
            word = nm.normalize_plain_text(match.group(1))
            if not word or word.startswith("-"):
                continue
            after = nm.normalize_plain_text(raw[match.end():match.end() + 140]).casefold()
            if re.match(source_pattern, after) and re.search(result_pattern, after):
                bases.append(word)
        plain = nm.normalize_plain_text(raw).casefold()
        for match in re.finditer(
            rf"\b(?:a|az)\s+([{word_chars}]+)(?:-|\s+)(?:ból|ből|{'|'.join(source_nouns)})\b"
            rf".{{0,120}}?{result_pattern}", plain):
            word = nm.normalize_plain_text(match.group(1))
            if word and word not in {"szó", "tő", "szótő"}:
                bases.append(word)
        for match in re.finditer(
            rf"{result_pattern}.{{0,120}}?\b(?:a|az)\s+([{word_chars}]+)(?:-|\s+)"
            rf"(?:ból|ből|{'|'.join(source_nouns)})\b", plain):
            word = nm.normalize_plain_text(match.group(1))
            if word and word not in {"szó", "tő", "szótő"}:
                bases.append(word)
    return list(dict.fromkeys(bases))


_DERIVATION_LIKE_TYPES = {
    "derivation", "back_formation", "conversion",
    "fossilization", "lexical_split", "coinage",
}
_COMPOUND_LIKE_TYPES = {
    "compound", "phrasal_compound", "constituent_lexicalization",
}

_HOMOGRAPH_REGEX = re.compile(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+$')



def _base_match_info(base: str, lemma: str) -> tuple[tuple[int, int], int] | None:
    """Return best match span plus quality (2 exact, 1 stem, 0 approximate)"""
    if not base or not lemma:
        return None
    b = _ascii_skeleton(_HOMOGRAPH_REGEX.sub("", base))
    l = _ascii_skeleton(_HOMOGRAPH_REGEX.sub("", lemma))
    if not b or not l:
        return None
    idx = l.find(b)
    if idx >= 0:
        return ((idx, idx + len(b)), 2)
    if len(b) >= 4:
        stem = b[:-1]
        idx = l.find(stem)
        if idx >= 0:
            return ((idx, idx + len(stem)), 1)
    best: tuple[int, int] | None = None
    best_score = 0.0
    min_len = max(2, len(b) - 2)
    max_len = min(len(l), len(b) + 2)
    for window_len in range(min_len, max_len + 1):
        for start in range(0, len(l) - window_len + 1):
            window = l[start:start + window_len]
            score = SequenceMatcher(None, b, window).ratio()
            if score > best_score:
                best_score = score
                best = (start, start + window_len)
    if best is not None and best_score >= 0.72:
        return (best, 0)
    return None


def _base_match_span(base: str, lemma: str) -> tuple[int, int] | None:
    info = _base_match_info(base, lemma)
    if info is None:
        return None
    return info[0]


_MIN_DERIVATION_SIMILARITY = 0.4 # --> ARBITRARY SIMILARITY PROPORTION


def _select_best_derivation_base(bases: list[str], lemma: str) -> list[str]:
    if not bases:
        return []
    ranked = sorted(
        bases,
        key=lambda b: (
            _lemma_word_similarity(lemma, b),
            len(_ascii_skeleton(_HOMOGRAPH_REGEX.sub("", b))),
        ),
        reverse=True,
    )
    best = ranked[0]
    if _lemma_word_similarity(lemma, best) < _MIN_DERIVATION_SIMILARITY:
        return []
    return [best]


def _select_compound_bases(bases: list[str], lemma: str) -> list[str]:
    """Pick the smallest useful set of parts that best covers the lemma"""
    if not bases or not lemma:
        return []
    lemma_skel = _ascii_skeleton(_HOMOGRAPH_REGEX.sub("", lemma))
    if not lemma_skel:
        return []

    candidates: list[tuple[str, tuple[int, int], int, float]] = []
    for base in bases:
        info = _base_match_info(base, lemma)
        if info is None:
            continue
        span, quality = info
        candidates.append((base, span, quality, _lemma_word_similarity(lemma, base)))
    if not candidates:
        return []

    candidates.sort(
        key=lambda item: (item[1][0], -item[2], -(item[1][1] - item[1][0]), -item[3])
    )
    covered = [False] * len(lemma_skel)
    selected: list[str] = []

    def uncovered_gain(span: tuple[int, int]) -> int:
        start, end = span
        return sum(1 for i in range(start, end) if not covered[i])

    def overlap_size(span: tuple[int, int]) -> int:
        start, end = span
        return sum(1 for i in range(start, end) if covered[i])

    for base, span, _quality, _score in candidates:
        gain = uncovered_gain(span)
        overlap = overlap_size(span)
        if not selected:
            selected.append(base)
            for i in range(span[0], span[1]):
                covered[i] = True
            continue
        if _quality == 0 and overlap > 0:
            continue
        if gain >= 2:
            selected.append(base)
            for i in range(span[0], span[1]):
                covered[i] = True
        if sum(covered) >= len(lemma_skel) - 1:
            break

    if not selected:
        selected.append(candidates[0][0])

    selected_spans = {_base_match_span(base, lemma): base for base in selected}
    remainder_parts = [
        base for base, span, _quality, _score in candidates
        if span not in selected_spans and uncovered_gain(span) >= 2
    ]
    for base in remainder_parts:
        span = _base_match_span(base, lemma)
        if span is None:
            continue
        selected.append(base)
        for i in range(span[0], span[1]):
            covered[i] = True
        if sum(covered) >= len(lemma_skel) - 1:
            break

    return list(dict.fromkeys(selected))


# Split markers that separate the actual derivation base from analogous comparisons.
# "vö." = compare
_VOE_SPLIT_REGEX = re.compile(r'\bvö\.|\bHasonló', re.IGNORECASE)


def _extract_formation_bases(etym: dict[str, Any], formation_type: str | None = None) -> list[str]:

    blocks = _collect_etymology_blocks(etym)
    formation_blocks = [
        b for b in blocks
        if b.get("role") == "in_hungarian"
    ]
    if not formation_blocks:
        return []
    if formation_type in _COMPOUND_LIKE_TYPES:
        return _italic_words_from_blocks(formation_blocks)
    full_text = " ".join(
        b.get("text_plain") or b.get("text") or b.get("text_raw") or "" for b in formation_blocks
    )
    if formation_type in _DERIVATION_LIKE_TYPES:
        strong_bases = _strong_italic_formation_bases(formation_blocks)
        if strong_bases:
            return strong_bases
    # Prefer references from before the first 'vö.', post-vö. items are analogies
    has_voe_marker = _VOE_SPLIT_REGEX.search(full_text) is not None
    pre_voe = _VOE_SPLIT_REGEX.split(full_text, maxsplit=1)[0]
    bases = list(dict.fromkeys(_extract_arrow_references(pre_voe)))
    if not bases:
        bases = list(dict.fromkeys(_extract_arrow_references(full_text)))
    return bases


def _set_formation_bases(target: dict[str, Any],
    etym: dict[str, Any], formation_type: str | None) -> None:

    bases = _extract_formation_bases(etym, formation_type)
    if not bases:
        return

    lemma = target.get("lemma") or ""
    lemma_skel = _ascii_skeleton(_HOMOGRAPH_REGEX.sub("", lemma)) if lemma else ""

    if lemma_skel and target.get("homograph") is None:
        bases = [
            b for b in bases
            if _ascii_skeleton(_HOMOGRAPH_REGEX.sub("", b)) != lemma_skel
        ]

    if not bases:
        return

    """ CHOOSING THE FORMATION TYPE, IMPORTANT STEP """
    if formation_type in _DERIVATION_LIKE_TYPES:

        bases = _select_best_derivation_base(bases, lemma) if lemma else bases[:1]

    elif formation_type in _COMPOUND_LIKE_TYPES:

        bases = _select_compound_bases(bases, lemma) if lemma else bases[:2]

    else:

        if formation_type == "simplex" and target.get("origin_type") == "internal":
            formation_type = None

        if lemma and len(bases) > 1:
            bases = (
                _select_compound_bases(bases, lemma)
                or _select_best_derivation_base(bases, lemma)
            )

    if not bases:
        return

    target.pop("formation_type", None)
    target["formation_type"] = formation_type

    if formation_type in _DERIVATION_LIKE_TYPES:
        target["derivation_base"] = bases[0]

    elif formation_type in _COMPOUND_LIKE_TYPES:
        target["compound_parts"] = bases

    elif not formation_type or formation_type in {"other", None}:

        if len(bases) == 1:
            target["derivation_base"] = bases[0]
        elif len(bases) > 1:
            target["compound_parts"] = bases[:2]



def _build_hypothesis_metadata(etym: dict[str, Any]) -> list[dict[str, Any]]:
    hypotheses = []
    summary_origin_type = _infer_origin_type(etym)

    for analysis in etym.get("analyses") or []:
        blocks = analysis.get("blocks") or []
        label_raw = analysis.get("label_raw") or ""
        text = _analysis_text(analysis)
        donor_blocks = [b for b in blocks if b.get("role") == "donor_data"]
        donor_text = " ".join(b.get("text_plain") or "" for b in donor_blocks)
        donor_words = []
        for block in donor_blocks:
            donor_words.extend(_language_word_pairs_from_block(block.get("text_raw") or ""))
        donor_langs = list(dict.fromkeys(
            pair["language"]
            for pair in donor_words
            if pair.get("language") and pair.get("language") != "magyar"
        ))
        mention_langs = [
            lang for lang in _find_language_mentions(label_raw)
            if lang != "magyar"
        ]
        if not donor_langs:
            donor_langs = [
                lang for lang in _find_language_mentions(donor_text)
                if lang != "magyar"
            ]
        langs = donor_langs + [lang for lang in mention_langs if lang not in donor_langs]
        primary_lang = langs[0] if langs else None
        origin_type = _infer_origin_from_label(label_raw, blocks)
        if summary_origin_type == "loan" and donor_blocks:
            origin_type = "loan"
        if origin_type is None and primary_lang and donor_blocks:
            origin_type = "loan"
        origin_type = _resolve_origin_type_with_family(origin_type, primary_lang)
        formation_type = _infer_formation_from_label(label_raw, origin_type)

        hypotheses.append({
            "analysis_index": analysis.get("index"),
            "label_plain": analysis.get("label_plain"),
            "origin_type": _hypothesis_origin_type(origin_type, formation_type, primary_lang),
            "formation_type": formation_type,
            "primary_donor_language": _english_language_name(primary_lang),
            "primary_donor_family": _language_family(primary_lang),
            "other_donor_languages": _english_language_names(langs[1:]),
            "corresponding_donor_word": _joined_donor_words(donor_blocks[0].get("text_raw") if donor_blocks else ""),
            "cross_referenced_bases": _extract_cross_references(text),
        })

    return hypotheses


def _parallel_donor_word_for_subentry(sub: dict[str, Any],
    etym: dict[str, Any], donor_block: dict[str, str] | None) -> str | None:

    if not donor_block:
        return None
    lemmas = _extract_bold_italic_lemmas(etym.get("summary_raw") or "")
    lemma = sub.get("lemma")
    if not lemmas or not lemma:
        return None

    lemma_key = nm.normalize_plain_text(lemma).casefold()
    lemma_keys = [nm.normalize_plain_text(item).casefold() for item in lemmas]
    if lemma_key not in lemma_keys:
        return None
    lemma_index = lemma_keys.index(lemma_key)

    donor_raw = donor_block.get("text_raw") or ""
    donor_parts = re.split(r'\s+\|\s+', donor_raw)
    if lemma_index >= len(donor_parts):
        return None
    return _first_italic_word(donor_parts[lemma_index])


def _joined_donor_words(text_raw: str) -> str | None:
    donor_words = _language_word_pairs_from_block(text_raw or "")
    unique = [pair["word"] for pair in donor_words if pair.get("word")]
    unique = list(dict.fromkeys(unique))
    return "|".join(unique) if unique else _first_italic_word(text_raw)


def _join_unique(values: list[str | None]) -> str | None:
    unique = [v for v in dict.fromkeys(values) if v]
    return "|".join(unique) if unique else None


def _join_hypothesis_values(values: list[Any]) -> str | None:
    """Join hypothesis metadata, preserving mixed missing/non-missing slots"""
    unique = list(dict.fromkeys(
        "None" if value in (None, "", []) else str(value)
        for value in values
    ))
    return "|".join(unique) if unique and unique != ["None"] else None


def _update_disputed_flags_from_hypotheses(etym: dict[str, Any]) -> None:
    hypotheses = etym.get("origin_hypotheses") or []
    if not hypotheses:
        return

    origin_values = {
        str(value)
        for value in (h.get("origin_type") for h in hypotheses)
        if value not in (None, "", [])
    }
    formation_values = {
        str(value)
        for value in (h.get("formation_type") for h in hypotheses)
        if value not in (None, "", [])
    }

    if len(origin_values) > 1:
        etym["is_disputed_any"] = True
        etym["is_disputed_origin"] = True
    if len(formation_values) > 1:
        etym["is_disputed_any"] = True
        etym["is_disputed_formation"] = True


def _extract_bold_italic_lemmas(text: str) -> list[str]:
    return list(dict.fromkeys(
        nm.normalize_plain_text(m.group(1)).strip()
        for m in re.finditer(r'<b>\s*<i>\s*([^<]+?)\s*</i>\s*</b>', text or "")
        if m.group(1).strip()
    ))


def _add_analysis_lemma_targets(etym: dict[str, Any]) -> None:
    for analysis in etym.get("analyses") or []:
        label_raw = analysis.get("label_raw") or ""
        body_raw = " ".join(block.get("text_raw") or "" for block in analysis.get("blocks") or [])
        analysis["target_lemmas"] = _extract_bold_italic_lemmas(label_raw)
        analysis["referenced_lemmas"] = _extract_bold_italic_lemmas(label_raw + " " + body_raw)


def _clean_empty_etymology_fields(etym: dict[str, Any]) -> None:
    for key in ["blocks", "analyses", "shared_blocks"]:
        if key in etym and not etym[key]:
            etym.pop(key, None)
    for key in ["other_text_raw", "other_text_plain", "other_etym_text_raw", "other_etym_text_plain"]:
        if key in etym and not etym[key]:
            etym.pop(key, None)


def _parse_family_etymology_chunk(chunk: dict[str, str]) -> dict[str, Any]:
    etym = {
        "shared_etym_sum_raw": chunk["raw_text"],
        "shared_etym_sum_plain": chunk["plain_text"],
    }
    parsed = parse_etymology_with_analyses(chunk["raw_text"])
    etym.update({
        "summary_raw": parsed.get("summary_raw", ""),
        "summary_plain": parsed.get("summary_plain", ""),
        "blocks": parsed.get("blocks", []),
        "analyses": parsed.get("analyses", []),
        "shared_blocks": parsed.get("shared_blocks", []),
        "other_etym_text_raw": parsed.get("other_text_raw", ""),
        "other_etym_text_plain": parsed.get("other_text_plain", ""),
        "etym_sum_raw": chunk["raw_text"],
    })
    _add_analysis_lemma_targets(etym)
    sn.mark_disputed_from_normalized_summary(etym)
    etym.pop("etym_sum_raw", None)
    etym["origin_hypotheses"] = _build_hypothesis_metadata(etym)
    for hypothesis, analysis in zip(etym.get("origin_hypotheses", []), etym.get("analyses", [])):
        hypothesis["target_lemmas"] = analysis.get("target_lemmas", [])
        hypothesis["referenced_lemmas"] = analysis.get("referenced_lemmas", [])
    _update_disputed_flags_from_hypotheses(etym)
    if etym.get("origin_hypotheses"):
        hypotheses = etym.get("origin_hypotheses", [])
        etym["origin_type"] = _join_unique([h.get("origin_type") for h in hypotheses])
        etym["formation_type"] = _join_unique([h.get("formation_type") for h in hypotheses])
        etym["primary_donor_language"] = _join_hypothesis_values(
            [h.get("primary_donor_language") for h in hypotheses]
        )
        etym["primary_donor_family"] = _join_hypothesis_values(
            [h.get("primary_donor_family") for h in hypotheses]
        )
        if etym.get("origin_type") in {"wanderword", "international"}:
            etym["primary_donor_family"] = etym["origin_type"]
    else:
        _update_direct_family_etymology_metadata(etym)
    _clean_empty_etymology_fields(etym)
    return etym


def _update_direct_family_etymology_metadata(etym: dict[str, Any]) -> None:
    """Infer metadata for one shared, unnumbered family etymology block"""
    blocks = _collect_etymology_blocks(etym)
    donor_blocks = [b for b in blocks if b.get("role") == "donor_data"]
    donor_text = " ".join(b.get("text_plain") or "" for b in donor_blocks)
    summary_raw = etym.get("summary_raw") or ""

    langs = _find_language_mentions(summary_raw + " " + donor_text)
    langs = [lang for lang in langs if lang != "magyar"]  # "magyar" is the target, never a donor
    primary_lang = langs[0] if langs else None

    origin_type = _infer_origin_type(etym)
    if (
        origin_type is None
        and primary_lang
        and donor_blocks
        and not _summary_implies_internal_model_based_formation(etym)
    ):
        origin_type = "loan"
    origin_type = _resolve_origin_type_with_family(origin_type, primary_lang)

    formation_type = _infer_formation_type(etym)
    if _mentions_folk_etymological_loanblend(etym) and formation_type in _COMPOUND_LIKE_TYPES:
        origin_type = "loanblend"
    if origin_type is None and formation_type in {
        "derivation",
        "compound",
        "phrasal_compound",
        "conversion",
        "back_formation",
        "reduplication",
        "fossilization",
        "constituent_lexicalization",
        "lexical_split",
        "vowel_harmony_shift",
        "semantic_split",
        "coinage",
        "language_reform",
    }:
        origin_type = "internal"
    if formation_type is None and origin_type in {
        "loan",
        "inherited",
        "onomatopoetic",
        "international",
        "wanderword",
        "unknown",
        "uncertain",
    }:
        formation_type = "simplex"
    if origin_type == "fictional" and formation_type in {None, "simplex"}:
        formation_type = "derivation"
    formation_type = _resolve_parser_formation_type(
        etym=etym,
        formation_type=formation_type,
        cross_referenced_bases=[],
    )
    block_hint = _non_native_hint_from_blocks(
        etym,
        origin_type,
        formation_type,
        etym.get("lemma"),
    )
    if block_hint:
        origin_type = block_hint.get("origin_type") or origin_type
        primary_lang = block_hint.get("primary_donor_language") or primary_lang
        langs = [primary_lang] + [lang for lang in block_hint.get("other_donor_languages", []) if lang != primary_lang]
    level, label = _infer_certainty(summary_raw, etym, origin_type)

    etym["origin_type"] = origin_type
    etym["formation_type"] = formation_type
    _set_formation_bases(etym, etym, formation_type)
    etym["origin_certainty_level"] = level
    etym["certainty_label"] = label
    if origin_type in {"internal", "coinage"}:
        primary_lang = None
        langs = []
    etym["primary_donor_language"] = _english_language_name(primary_lang)
    donor_family = _language_family(primary_lang)
    if donor_family is None and re.search(r'\balapnyelvi\b', summary_raw, re.IGNORECASE):
        donor_family = "Uralic"
    if origin_type in {"wanderword", "international"}:
        donor_family = origin_type
    etym["primary_donor_family"] = donor_family
    etym["other_donor_languages"] = _english_language_names(langs[1:])
    etym["corresponding_donor_word"] = (
        (block_hint or {}).get("corresponding_donor_word")
        or _first_italic_word(donor_blocks[0].get("text_raw") if donor_blocks else "")
    )
    etym["related_words"] = _extract_related_words(blocks)
    secondary_hypotheses = _secondary_origin_hypotheses_from_blocks(etym, origin_type)
    if secondary_hypotheses:
        etym["secondary_origin_hypotheses"] = secondary_hypotheses


def _mentioned_subentry_indices(text: str, subentries: list[dict[str, Any]]) -> list[int]:
    plain = nm.normalize_plain_text(text or "").casefold()
    bold_italic_mentions = {
        lemma.casefold()
        for lemma in _extract_bold_italic_lemmas(text or "")
    }
    matches = []
    for sub in subentries:
        lemma = sub.get("lemma")
        index = sub.get("subentry_index")
        if not lemma or index is None:
            continue
        lemma_plain = nm.normalize_plain_text(lemma).casefold()
        if lemma_plain in bold_italic_mentions or re.search(rf'(?<!\w){re.escape(lemma_plain)}(?!\w)', plain):
            matches.append(index)
    return list(dict.fromkeys(matches))


def _explicit_targeted_subentry_indices(chunk_raw: str, chunk_plain: str,
    subentries: list[dict[str, Any]],) -> list[int]:

    plain = nm.normalize_plain_text(chunk_plain or chunk_raw or "").casefold()
    if not plain:
        return []

    matches: list[int] = []
    for sub in subentries:
        lemma = sub.get("lemma")
        index = sub.get("subentry_index")
        if not lemma or index is None:
            continue

        lemma_plain = nm.normalize_plain_text(lemma).casefold()
        patterns = [
            rf'(?<!\w)a\s+{re.escape(lemma_plain)}\s+származékszó\b',
            rf'(?<!\w)a\s+{re.escape(lemma_plain)}\s+analógiás alakulat\b',
            rf'(?<!\w)a\s+{re.escape(lemma_plain)}\s+lexikai hasadás\b',
            rf'(?<!\w)szócsalád alapja,\s*a\s+{re.escape(lemma_plain)}\b',
            rf'(?<!\w)szócsalád legkorábbi tagja,\s*a\s+{re.escape(lemma_plain)}\b',
            rf'(?<!\w)szócsalád korábbi tagja,\s*a\s+{re.escape(lemma_plain)}\b',
        ]
        if any(re.search(pattern, plain) for pattern in patterns):
            matches.append(index)

    return list(dict.fromkeys(matches))


def _copy_etymology_chunk_to_subentry(sub: dict[str, Any], chunk: dict[str, str]) -> None:
    etym = sub["subentry_etymology"]
    etym["etym_sum_raw"] = chunk["raw_text"]
    etym["etym_sum_plain"] = chunk["plain_text"]
    etym["blocks"] = bx.extract_symbolic_etymology_blocks(chunk["raw_text"])


def _copy_family_hypotheses_to_target_subentries(entry: dict[str, Any]) -> None:
    family_etym = entry.get("family_etymology") or {}
    hypotheses = family_etym.get("origin_hypotheses") or []
    if not hypotheses:
        return

    subentries = entry.get("subentries") or []
    lemma_to_subs: dict[str, list[dict[str, Any]]] = {}
    for sub in subentries:
        lemma = sub.get("lemma")
        if not lemma:
            continue
        lemma_to_subs.setdefault(nm.normalize_plain_text(lemma).casefold(), []).append(sub)

    for hypothesis in hypotheses:
        for target in hypothesis.get("target_lemmas") or []:
            target_key = nm.normalize_plain_text(target).casefold()
            for sub in lemma_to_subs.get(target_key, []):
                sub["corresponding_donor_word"] = hypothesis.get("corresponding_donor_word")
                sub["origin_type"] = hypothesis.get("origin_type")
                sub["formation_type"] = hypothesis.get("formation_type")
                sub["primary_donor_language"] = hypothesis.get("primary_donor_language")
                sub["primary_donor_family"] = hypothesis.get("primary_donor_family")
                sub["other_donor_languages"] = hypothesis.get("other_donor_languages", [])
                sub["cross_referenced_bases"] = hypothesis.get("cross_referenced_bases", [])
                sub["origin_hypotheses"] = [hypothesis]
                sub["formation_hypotheses"] = [
                    {
                        "analysis_index": hypothesis.get("analysis_index"),
                        "formation_type": hypothesis.get("formation_type"),
                        "cross_referenced_bases": hypothesis.get("cross_referenced_bases", []),
                    }
                ]
                sub["related_words"] = list(dict.fromkeys(
                    (sub.get("related_words") or [])
                    + (hypothesis.get("cross_referenced_bases") or [])
                ))

                if family_etym.get("is_disputed_any"):
                    sub["origin_certainty_level"] = 1
                    sub["certainty_label"] = "uncertain"


def _copy_shared_family_etymology_to_subentries(entry: dict[str, Any]) -> None:

    family_etym = entry.get("family_etymology") or {}
    if family_etym.get("origin_hypotheses") and not (family_etym.get("origin_type") or family_etym.get("formation_type")):
        return
    if not family_etym.get("origin_type") and not family_etym.get("formation_type"):
        return

    subentries = entry.get("subentries") or []
    base_indices = set(_earliest_subentry_indices(subentries))
    base_lemmas = {nm.normalize_plain_text(sub.get("lemma") or "").casefold()
        for sub in subentries
        if sub.get("subentry_index") in base_indices}
    base_indices.update(
        sub.get("subentry_index")
        for sub in subentries
        if nm.normalize_plain_text(sub.get("lemma") or "").casefold() in base_lemmas)
    summary = nm.normalize_plain_text((family_etym.get("summary_raw") or "") or (family_etym.get("summary_plain") or "")).casefold()
    productive_suffix_formation = _mentions_productive_suffix_formation(family_etym)
    inherited_stem_with_hungarian_suffix = (productive_suffix_formation
        and family_etym.get("origin_type") == "inherited"
        and "tő magyar képzéssel" in summary)
    split_family_formation = (family_etym.get("formation_type") == "derivation"
        or productive_suffix_formation
        or ("jövevényszó" in summary
            and "jövevényszók" not in summary
            and "jövevényszav" not in summary))

    all_are_derivatives = ("származékszók" in summary
        or "származékszavak" in summary
        or (productive_suffix_formation
            and family_etym.get("origin_type") in {"fictional", "onomatopoetic"})
        or inherited_stem_with_hungarian_suffix)

    if (family_etym.get("formation_type") == "simplex"
        and family_etym.get("origin_type") != "loan"):
        split_family_formation = False
        all_are_derivatives = False
    if (family_etym.get("formation_type") == "derivation"
        and family_etym.get("origin_type") in {"fictional", "onomatopoetic"}
        and ("származékszó" in summary
            or "szÃ¡rmazÃ©kszÃ³" in summary) # in case the character has not been recovered
        and not any(marker in summary for marker in (
            "származékszók",
            "származékszav",
            "származékok",
            "szÃ¡rmazÃ©kszÃ³k",
            "szÃ¡rmazÃ©kszav",
            "szÃ¡rmazÃ©kok",))):
        all_are_derivatives = False
    if (family_etym.get("formation_type") == "derivation"
        and family_etym.get("origin_type") == "internal"
        and family_etym.get("derivation_base")):
        all_are_derivatives = True

    inherited_fields = [
        "origin_type",
        "origin_certainty_level",
        "certainty_label",
        "primary_donor_language",
        "primary_donor_family",
        "other_donor_languages",
        "corresponding_donor_word",
        "related_words"]

    for sub in subentries:
        sub_etym = sub.get("subentry_etymology") or {}
        if sub_etym.get("summary_raw") or sub_etym.get("summary_plain"):
            continue
        for field in inherited_fields:
            if family_etym.get(field) not in (None, [], ""):
                sub[field] = family_etym.get(field)
        if split_family_formation:
            if all_are_derivatives:
                sub["formation_type"] = "derivation"
            else:
                sub["formation_type"] = (
                    "simplex"
                    if sub.get("subentry_index") in base_indices
                    else "derivation")
        elif family_etym.get("formation_type"):
            sub["formation_type"] = family_etym.get("formation_type")
        named_as_hungarian_derivative = _lemma_is_named_as_hungarian_derivative(
            sub.get("lemma"),
            family_etym)
        if (productive_suffix_formation
            and sub.get("formation_type") == "simplex"
            and named_as_hungarian_derivative):
            sub["formation_type"] = "derivation"

    family_base = family_etym.get("derivation_base")
    if family_etym.get("origin_type") == "internal" and family_base:
        family_base_skel = _ascii_skeleton(_HOMOGRAPH_REGEX.sub("", family_base))
        for sub in subentries:
            if sub.get("derivation_base") or sub.get("compound_parts"):
                continue
            if sub.get("formation_type") != "derivation":
                continue
            sub_etym = sub.get("subentry_etymology") or {}
            if (sub_etym.get("summary_raw")
                or sub_etym.get("summary_plain")
                or sub_etym.get("blocks")
                or sub_etym.get("analyses")):
                continue
            lemma = sub.get("lemma") or ""
            lemma_skel = _ascii_skeleton(_HOMOGRAPH_REGEX.sub("", lemma)) if lemma else ""
            if lemma_skel and family_base_skel and lemma_skel == family_base_skel and sub.get("homograph") is None:
                continue
            sub["derivation_base"] = family_base
    return

def _adjust_untargeted_family_etymology_formations(entry: dict[str, Any]) -> None:

    flagged = [sub for sub in entry.get("subentries") or []
        if sub.pop("_untargeted_family_etymology", False)]
    if not flagged:
        return

    base_indices = set(_earliest_subentry_indices(flagged))
    for sub in flagged:
        sub_etym = sub.get("subentry_etymology") or {}
        all_are_derivatives = (
            _mentions_derivative_word(sub_etym.get("summary_raw") or sub_etym.get("summary_plain") or sub_etym.get("summary") or "")
            or (_mentions_productive_suffix_formation(sub_etym)
                and sub.get("origin_type") in {"fictional", "onomatopoetic"}))
        sub["formation_type"] = ("derivation"
            if all_are_derivatives or sub.get("subentry_index") not in base_indices
            else "simplex")


def _earliest_subentry_indices(subentries: list[dict[str, Any]]) -> list[int]:
    """Return indices of the earliest attested family members"""
    dated = [
        (sub.get("first_year_of_attestation"), sub.get("subentry_index"))
        for sub in subentries
        if sub.get("first_year_of_attestation") is not None
        and sub.get("subentry_index") is not None]
    if not dated:
        return [subentries[0]["subentry_index"]] if subentries else []
    first_year = min(year for year, _ in dated)
    return [index for year, index in dated if year == first_year]


def _flatten_single_entry(entry: dict[str, Any]) -> None:
    subentries = entry.pop("subentries", [])
    sub = subentries[0] if subentries else {}

    entry.pop("number_of_entries", None)
    entry.pop("base_of_word_family", None)
    entry.pop("family_etymology", None)

    for key, value in sub.items():
        if key == "subentry_index":
            continue
        entry[key] = value


def _update_etymology_metadata(sub: dict[str, Any]) -> None:
    etym = sub.get("subentry_etymology", {})
    blocks = _collect_etymology_blocks(etym)
    donor_blocks = [b for b in blocks if b.get("role") == "donor_data"]
    donor_text = " ".join(b.get("text_plain") or "" for b in donor_blocks)

    langs = _find_language_mentions((etym.get("summary_raw") or "") + " " + donor_text)
    langs = [lang for lang in langs if lang != "magyar"]  # "magyar" is  never a donor
    primary_lang = langs[0] if langs else None
    origin_type = _infer_origin_type(etym)
    if (origin_type is None
        and primary_lang
        and donor_blocks
        and not _summary_implies_internal_model_based_formation(etym)):
        origin_type = "loan"
    origin_type = _resolve_origin_type_with_family(origin_type, primary_lang)
    formation_type = _infer_formation_type(etym)
    if _mentions_folk_etymological_loanblend(etym) and formation_type in _COMPOUND_LIKE_TYPES:
        origin_type = "loanblend"
    if origin_type is None and formation_type in {
        "derivation",
        "compound",
        "phrasal_compound",
        "conversion",
        "back_formation",
        "reduplication",
        "fossilization",
        "constituent_lexicalization",
        "lexical_split",
        "vowel_harmony_shift",
        "semantic_split",
        "coinage",
        "language_reform"}:
        origin_type = "internal"
    if formation_type is None and origin_type in {
        "loan",
        "inherited",
        "onomatopoetic",
        "international",
        "wanderword",
        "unknown",
        "uncertain"}:
        formation_type = "simplex"
    if origin_type == "fictional" and formation_type in {None, "simplex"}:
        formation_type = "derivation"
    block_hint = _non_native_hint_from_blocks(
        etym,
        origin_type,
        formation_type,
        sub.get("lemma"))
    if block_hint:
        origin_type = block_hint.get("origin_type") or origin_type
        primary_lang = block_hint.get("primary_donor_language") or primary_lang
        langs = [primary_lang] + [lang for lang in block_hint.get("other_donor_languages", []) if lang != primary_lang]
    level, label = _infer_certainty(etym.get("summary_raw") or "", etym, origin_type)

    _set_first_year_from_attestations(sub)
    sub["origin_type"] = origin_type
    sub["formation_type"] = formation_type
    _set_formation_bases(sub, etym, formation_type)
    sub["origin_certainty_level"] = level
    sub["certainty_label"] = label
    
    if origin_type in {"internal", "coinage"}:
        primary_lang = None
        langs = []
    sub["primary_donor_language"] = _english_language_name(primary_lang)
    donor_family = _language_family(primary_lang)
    if donor_family is None and re.search(r'\balapnyelvi\b', etym.get("summary_plain") or "", re.IGNORECASE):
        donor_family = "Uralic"
    if origin_type in {"wanderword", "international"}:
        donor_family = origin_type
    sub["primary_donor_family"] = donor_family
    sub["other_donor_languages"] = _english_language_names(langs[1:])
    sub["corresponding_donor_word"] = (
        (block_hint or {}).get("corresponding_donor_word")
        or
        _parallel_donor_word_for_subentry(sub, etym, donor_blocks[0] if donor_blocks else None)
        or _first_italic_word(donor_blocks[0].get("text_raw") if donor_blocks else "")
    )
    sub["related_words"] = _extract_related_words(blocks)

    hypotheses = _build_hypothesis_metadata(etym)
    if origin_type == "loan":
        for hypothesis in hypotheses:
            if hypothesis.get("primary_donor_family") or hypothesis.get("primary_donor_language"):
                hypothesis["origin_type"] = "loan"
    sub["origin_hypotheses"] = hypotheses
    sub["formation_hypotheses"] = [
        {
            "analysis_index": h.get("analysis_index"),
            "formation_type": h.get("formation_type"),
            "cross_referenced_bases": h.get("cross_referenced_bases", []),
        }
        for h in hypotheses
        if h.get("formation_type") or h.get("cross_referenced_bases")]
    sub["cross_referenced_bases"] = list(dict.fromkeys(
        word
        for h in hypotheses
        for word in h.get("cross_referenced_bases", [])))
    sub["formation_type"] = _resolve_parser_formation_type(
        etym=etym,
        formation_type=sub.get("formation_type"),
        cross_referenced_bases=sub.get("cross_referenced_bases"))
    secondary_hypotheses = _secondary_origin_hypotheses_from_blocks(etym, sub.get("origin_type"))
    if secondary_hypotheses:
        sub["secondary_origin_hypotheses"] = secondary_hypotheses

    if etym.get("is_disputed_any") and hypotheses:
        hypothesis_origins = _join_unique([h.get("origin_type") for h in hypotheses])
        hypothesis_formations = _join_unique([h.get("formation_type") for h in hypotheses])
        hypothesis_languages = _join_hypothesis_values([h.get("primary_donor_language") for h in hypotheses])
        hypothesis_families = _join_hypothesis_values([h.get("primary_donor_family") for h in hypotheses])
        hypothesis_donor_words = _join_unique([h.get("corresponding_donor_word") for h in hypotheses])
        sub["origin_type"] = (
            origin_type
            if origin_type and origin_type not in {"disputed", "unknown", "uncertain"}
            else (hypothesis_origins or origin_type))
        if hypothesis_formations:
            sub["formation_type"] = hypothesis_formations
        elif etym.get("is_disputed_formation"):
            sub["formation_type"] = "disputed"
        elif sub.get("origin_type") and any(
            origin in str(sub.get("origin_type")).split("|")
            for origin in ["loan", "inherited", "onomatopoetic", "international", "wanderword"]):
            sub["formation_type"] = formation_type or "simplex"
        else:
            sub["formation_type"] = formation_type
        sub["primary_donor_language"] = hypothesis_languages
        sub["primary_donor_family"] = (
            sub["origin_type"] if sub.get("origin_type") in {"wanderword", "international"}
            else hypothesis_families)
        if hypothesis_donor_words:
            sub["corresponding_donor_word"] = hypothesis_donor_words
        sub["formation_type"] = _resolve_parser_formation_type(
            etym=etym,
            formation_type=sub.get("formation_type"),
            cross_referenced_bases=sub.get("cross_referenced_bases"))


def _inherit_metadata_from_family_bases(entry: dict[str, Any]) -> None:

    subentries = entry.get("subentries") or []
    base_indices = set(entry.get("base_of_word_family") or [])
    if not subentries or not base_indices:
        return

    bases = [sub for sub in subentries
        if sub.get("subentry_index") in base_indices and sub.get("primary_donor_family")]
    if not bases:
        return
    base = bases[0]

    donor_fields = ["primary_donor_language",
        "primary_donor_family",
        "other_donor_languages",
        "corresponding_donor_word"]
    for sub in subentries:
        if sub in bases:
            continue
        if sub.get("primary_donor_family"):
            continue
        if sub.get("origin_type") in {"internal", "loan"}:
            for field in donor_fields:
                if base.get(field):
                    sub[field] = base.get(field)
            if (sub.get("origin_type") == "internal"
                and base.get("origin_type") not in {None, "", "internal"}):
                sub["origin_type"] = base.get("origin_type")




# ------------------- SYMBOL HELPERS ----------------------------------------------


def _identify_symbol(text: str) -> tuple[str | None, str | None, str]:

    stripped = text.lstrip()
    for sym in ETY_SYMBOLS_ORDERED:
        if stripped.startswith(sym):
            role = ab.ETY_BLOCK_ROLE_MAP.get(sym)
            rest = stripped[len(sym):].lstrip()
            return sym, role, rest
    return None, None, stripped


def extract_symbolic_blocks(etym_text: str) -> list[dict[str, str]]:

    if not etym_text:
        return []

    # split at symbol boundaries
    chunks = ETY_SYMBOL_SPLIT_RE.split(etym_text)
    blocks = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        sym, role, rest = _identify_symbol(chunk)
        if sym is None:

            if blocks:
                blocks[-1]["text_raw"] += " " + chunk
                blocks[-1]["text_plain"] = nm.normalize_plain_text(blocks[-1]["text_raw"])
            continue

        blocks.append({
            "symbol": sym,
            "role": role or "unknown",
            "text_raw": rest,
            "text_plain": nm.normalize_plain_text(rest),
        })

    return blocks

_NUMBERED_ANALYSIS_RE = re.compile(
    r'\|\s*(\d+)\.\s*'
)


def _normalize_numbered_analysis_markers(text: str) -> str:
    """
    Make numbered etymology alternatives split-friendly.
    The source commonly has alternatives as <b>1.</b> ... <b>2.</b>
    instead of literal "| 1." / "| 2." markers.
    """
    return re.sub(r'(?:\|\s*)?<b>\s*(\d+)\.\s*</b>\s*', r'| \1. ', text)


def _split_shared_etymology_tail(text: str) -> tuple[str, str]:
    """
    Split off a ▣ section, which applies to the whole etymology rather than to
    the preceding numbered analysis.
    """
    if "▣" not in text:
        return text, ""
    before, after = text.split("▣", 1)
    return before.strip(), after.strip()


def _split_after_summary(text_after_summary: str) -> list[dict]:

    text_after_summary = _normalize_numbered_analysis_markers(text_after_summary)

    splits = list(_NUMBERED_ANALYSIS_RE.finditer(text_after_summary))
    if not splits:
        return []

    analyses = []

    for i, m in enumerate(splits):
        idx = int(m.group(1))
        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text_after_summary)
        segment = text_after_summary[start:end].strip()

        if "|" in segment:
            label_part, body_part = segment.split("|", 1)
        else:
            label_part = segment
            body_part = ""

        analyses.append({
            "index": idx,
            "label_raw": label_part.strip(),
            "raw_body": body_part.strip()})

    return analyses


def parse_etymology_with_analyses(etym_sum_raw: str) -> dict[str, Any]:

    result = {
        "summary_raw": "",
        "summary_plain": "",
        "blocks": [],
        "analyses": [],
        "shared_blocks": [],
        "other_text_raw": "",
        "other_text_plain": ""}

    if not etym_sum_raw or not etym_sum_raw.strip():
        return result

    text = etym_sum_raw.strip()

    text_no_marker = re.sub(r'^\s*■\s*', '', text, count=1)


    if "|" not in text_no_marker:
        result["summary_raw"] = text_no_marker.strip()
        result["summary_plain"] = nm.normalize_plain_text(text_no_marker).rstrip(" .;:")
        return result

    summary_part, body_part = text_no_marker.split("|", 1)
    summary_raw = summary_part.strip()
    summary_plain = nm.normalize_plain_text(summary_raw).rstrip(" .;:")

    result["summary_raw"] = summary_raw
    result["summary_plain"] = summary_plain

    body_part = _normalize_numbered_analysis_markers(body_part)
    has_numbered = bool(re.search(r'(^|\|)\s*1\.', body_part))

    if has_numbered:
        raw_analyses = _split_after_summary("|" + body_part)

        for ra in raw_analyses:
            label_raw = ra["label_raw"]
            label_plain = nm.normalize_plain_text(label_raw).rstrip(" .;:")

            raw_body, shared_tail = _split_shared_etymology_tail(ra["raw_body"])
            blocks = extract_symbolic_blocks(raw_body)

            if shared_tail:
                result["shared_blocks"].extend(extract_symbolic_blocks(shared_tail))

            result["analyses"].append({
                "index": ra["index"],
                "label_raw": label_raw,
                "label_plain": label_plain,
                "blocks": blocks})

    else:

        body_part, shared_tail = _split_shared_etymology_tail(body_part)
        blocks = extract_symbolic_blocks(body_part)

        analysis_blocks = []
        shared_blocks = []

        SHARED_ROLES = {
            "belong_here",
            "belong_not_here",
            "belong_maybe_here",
            "rejected_etymology",
            "from_hungarian"}

        for b in blocks:
            if b["role"] in SHARED_ROLES:
                shared_blocks.append(b)
            else:
                analysis_blocks.append(b)

        result["blocks"] = analysis_blocks
        result["shared_blocks"] = shared_blocks

        if shared_tail:
            result["shared_blocks"].extend(extract_symbolic_blocks(shared_tail))

    return result


# ---------------------  ATTACH PARSED ETYMOLOGY TO A SUBENTRY


def enrich_subentry_etymology(sub: dict[str, Any]) -> None:

    etym = sub.get("subentry_etymology", {})
    raw = etym.get("etym_sum_raw", "")

    parsed = parse_etymology_with_analyses(raw)

    etym = {
        "summary_raw": parsed["summary_raw"],
        "summary_plain": parsed["summary_plain"],
        "blocks": parsed.get("blocks", []),
        "analyses": parsed.get("analyses", []),
        "shared_blocks": parsed.get("shared_blocks", []),
        "other_text_raw": parsed.get("other_text_raw", ""),
        "other_text_plain": parsed.get("other_text_plain", ""),
        "etym_sum_raw": raw,
    }

    sn.mark_disputed_from_normalized_summary(etym)
    etym.pop("etym_sum_raw", None)
    etym["origin_hypotheses"] = _build_hypothesis_metadata(etym)
    _update_disputed_flags_from_hypotheses(etym)

    if not etym["blocks"]:
        etym.pop("blocks", None)
    if not etym["analyses"]:
        etym.pop("analyses", None)
    if not etym["shared_blocks"]:
        etym.pop("shared_blocks", None)
    if not etym["other_text_raw"]:
        etym.pop("other_text_raw", None)
    if not etym["other_text_plain"]:
        etym.pop("other_text_plain", None)

    sub["subentry_etymology"] = etym
    sub["is_duplicate"] = bool(etym.get("is_duplicate"))
    _update_etymology_metadata(sub)




#___________________________________________________________________________#
# ------------------- MAIN DISPATCHER IN PIPELINE ----------------------- #

def extract_entry_block(raw_entry_text: str) -> dict[str, Any]:

    if not raw_entry_text or not raw_entry_text.strip():
        return _finalize_entry_output({
            "entry_type": "unknown",
            "raw_text": raw_entry_text or "",
            "plain_text": "",
            "parser_warnings": ["Empty entry block."],
        })

    # Normalize into line dicts: [{"raw": "...", "plain": "..."}, ...]
    lines = nm.normalize_entry_lines(raw_entry_text)

    if not lines:
        return _finalize_entry_output({
            "entry_type": "unknown",
            "raw_text": raw_entry_text,
            "plain_text": nm.normalize_plain_text(raw_entry_text),
            "parser_warnings": ["No parseable lines in entry block."]})

    first_plain = lines[0]["plain"].strip()

    if ep.is_redirect_line(first_plain):
        return _finalize_entry_output(eb.make_redirect_entry(raw_entry_text, lines[0]["raw"]))
 
    if ep.is_roman_grouped_tilde_entry(lines):
        entry = eb.make_roman_grouped_tilde_entry(raw_entry_text, lines)
        _enrich_affix_family_etymology(entry, lines)
        _fill_book_block(entry, lines)
        return _finalize_entry_output(entry)

    if ep.is_prefix_family_entry(lines):
        entry = eb.make_prefix_family_entry(raw_entry_text, lines)
        _enrich_affix_family_etymology(entry, lines)
        _fill_book_block(entry, lines)
        return _finalize_entry_output(entry)

    if ep.is_suffix_family_entry(lines):
        entry = eb.make_suffix_family_entry(raw_entry_text, lines)
        _enrich_suffix_family_entry(entry, lines)
        _fill_book_block(entry, lines)
        return _finalize_entry_output(entry)

    head_lines_raw = _collect_head_lines(lines)

    if not head_lines_raw:
        return _finalize_entry_output({
            "entry_type": "unknown",
            "raw_text": raw_entry_text,
            "plain_text": nm.normalize_plain_text(raw_entry_text),
            "parser_warnings": ["No headword line found."],
        })

    entry = eb.make_word_family_entry(raw_entry_text, head_lines_raw)

    if len(head_lines_raw) == 1:
        entry["entry_type"] = "single"
    else:
        entry["entry_type"] = "family"

    _fill_subentries(entry, lines)
    _attach_family_etymologies(entry, lines)
    _clean_entry_head_lines(entry)

    for sub in entry.get("subentries", []):
        enrich_subentry_etymology(sub)

    _fill_book_block(entry, lines)

    if entry["entry_type"] == "family":
        _copy_shared_family_etymology_to_subentries(entry)
        _copy_family_hypotheses_to_target_subentries(entry)
        _adjust_untargeted_family_etymology_formations(entry)

    entry["base_of_word_family"] = _determine_family_base(entry)
    if entry["entry_type"] == "family":
        _inherit_metadata_from_family_bases(entry)

    if entry["entry_type"] == "single":
        _flatten_single_entry(entry)

    return _finalize_entry_output(entry)



# ------------------ INTERNAL HELPERS FOR extract_entry_block

def _enrich_suffix_family_entry(entry: dict[str, Any], lines: list[dict[str, str]]) -> None:

    suffix = (entry.get("lemma") or "").lstrip("-")
    body_raw = _suffix_family_body_raw(lines)
    if not body_raw:
        return

    family_etym = _suffix_family_etymology(lines)
    if family_etym:
        entry["family_etymology"] = family_etym

    subentries: list[dict[str, Any]] = []
    roman_groups = []
    member_forms: list[str] = []

    for group in _split_suffix_body_into_roman_groups(body_raw):
        members = []
        for segment, head_raw, flags_raw in _suffix_member_segments(group["raw"]):
            head_plain = nm.normalize_plain_text(head_raw)
            lemma = _expand_suffix_member_form(head_plain, suffix)
            sub = eb.make_empty_subentry(len(subentries) + 1, head_raw)
            sub["lemma"] = lemma
            sub["head_line_plain"] = lemma
            sub["flags"]["is_obsolete"] = "†" in flags_raw
            sub["flags"]["is_archaic"] = "△" in flags_raw or "∆" in flags_raw
            sub["flags"]["is_dialectal"] = "×" in flags_raw

            _parse_subentry_zones(sub, [{"raw": segment, "plain": nm.normalize_plain_text(segment)}])
            sub["lemma"] = lemma
            sub["head_line_plain"] = lemma
            _set_first_year_from_attestations(sub)
            _copy_family_etymology_metadata_to_suffix_member(sub, family_etym)

            subentries.append(sub)
            members.append(lemma)
            member_forms.append(lemma)

        additional_members = _suffix_additional_members(group["raw"], suffix)
        for item in additional_members:
            member_forms.append(item["form"])

        group_obj = {"roman": group["roman"], "members": members}
        if additional_members:
            group_obj["additional_members"] = additional_members
        roman_groups.append(group_obj)

    entry["subentries"] = subentries
    entry["number_of_entries"] = len(subentries) or entry.get("number_of_entries", 1)
    entry["roman_groups"] = roman_groups
    entry["member_forms"] = list(dict.fromkeys(member_forms))


_AFFIX_FAMILY_TOP_LEVEL_FIELDS = (
    "origin_type",
    "formation_type",
    "origin_certainty_level",
    "certainty_label",
    "primary_donor_language",
    "primary_donor_family",
    "other_donor_languages",
    "corresponding_donor_word",
)


def _enrich_affix_family_etymology(entry: dict[str, Any], lines: list[dict[str, str]]) -> None:

    chunks = bx.extract_etymology_chunks(lines)
    if not chunks:
        return
    family_etym = _parse_family_etymology_chunk(chunks[0])
    entry["family_etymology"] = family_etym
    for field in _AFFIX_FAMILY_TOP_LEVEL_FIELDS:
        value = family_etym.get(field)
        if value is not None:
            entry[field] = value


def _suffix_family_body_raw(lines: list[dict[str, str]]) -> str:
    """Return the raw suffix-family body between the suffix head and etymology"""
    body = []
    head_found = False
    for ln in lines:
        plain = ln["plain"].strip()
        if not head_found:
            if rx.SUFFIX_FAMILY_HEAD_RE.match(plain):
                head_found = True
            continue
        if ep.is_etymology_start_line(plain) or ep.is_book_start_line(plain):
            break
        body.append(ln["raw"].strip())
    return re.sub(r"\s+", " ", " ".join(part for part in body if part)).strip()


def _split_suffix_body_into_roman_groups(body_raw: str) -> list[dict[str, str]]:
    """Split a suffix-family body into Roman numeral groups with raw text"""
    roman_re = re.compile(r'<b>\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*</b>')
    matches = list(roman_re.finditer(body_raw))
    groups = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body_raw)
        groups.append({
            "roman": match.group(1),
            "raw": body_raw[match.end():end].strip(),
        })
    return groups


def _suffix_member_segments(group_raw: str) -> list[tuple[str, str, str]]:

    start_re = re.compile(
        r'(?:<b>\s*\|\s*</b>\s*)?'
        r'(?P<head>(?:<b>\s*(?!A:|\|)[^<]+\s*</b>\s*){1,5})'
        r'(?P<flags>[†△×∆\s]*)'
        r'<b>\s*A:',
        re.DOTALL,
    )
    starts = list(start_re.finditer(group_raw))
    segments = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(group_raw)
        segment = group_raw[start.start():end].strip()
        segment = re.sub(r'^<b>\s*\|\s*</b>\s*', '', segment)
        segment = re.split(r'(?:<b>\s*\|\s*</b>\s*)?Ilyenek még:', segment, maxsplit=1)[0].strip()
        segments.append((segment, start.group("head").strip(), start.group("flags") or ""))
    return segments


def _expand_suffix_member_form(form: str, suffix: str) -> str:
    """Expand suffix placeholder notation such as 'jó~' with the suffix body"""
    form = form.strip()
    if "~" not in form:
        return form
    return form.replace("~", suffix)


def _suffix_additional_members(group_raw: str, suffix: str) -> list[dict[str, Any]]:
    """Parse lightweight 'Ilyenek még:' examples from one Roman group"""
    plain = nm.normalize_plain_text(group_raw)
    match = re.search(r'Ilyenek még:\s*(.*)$', plain)
    if not match:
        return []

    items = []
    for part in _split_top_level_semicolon(match.group(1)):
        part = part.strip().strip(";")
        item_match = re.match(
            r"(?P<form>\S+)\s+(?P<meaning>'[^']+')\s*(?P<source>\([^)]*\))?",
            part)
        if not item_match:
            continue
        source = item_match.group("source") or ""
        year_match = re.search(r'\b(\d{3,4})\b', source)
        item: dict[str, Any] = {
            "form": _expand_suffix_member_form(item_match.group("form"), suffix),
            "meaning": item_match.group("meaning")}
        if year_match:
            item["year"] = int(year_match.group(1))
        if source:
            item["source"] = source
        items.append(item)
    return items


def _suffix_family_etymology(lines: list[dict[str, str]]) -> dict[str, Any]:
    """Parse the shared suffix-family etymology chunk, if present"""
    chunks = bx.extract_etymology_chunks(lines)
    return _parse_family_etymology_chunk(chunks[0]) if chunks else {}


def _copy_family_etymology_metadata_to_suffix_member(sub: dict[str, Any], family_etym: dict[str, Any]) -> None:
    """Copy shared suffix-family origin metadata onto one parsed member"""
    if not family_etym:
        return
    for field in [
        "origin_type",
        "formation_type",
        "origin_certainty_level",
        "certainty_label",
        "primary_donor_language",
        "primary_donor_family",
        "other_donor_languages",
        "corresponding_donor_word",
        "related_words"]:
        value = family_etym.get(field)
        if value:
            sub[field] = value


def _finalize_entry_output(entry: dict[str, Any]) -> dict[str, Any]:
    """Remove redundant raw mirrors from the public JSON shape"""
    raw_text = entry.get("raw_text", "")
    if (entry.get("family_etymology") or {}).get("is_duplicate"):
        entry["is_duplicate"] = True
    cleaned = _strip_redundant_raw_fields(entry, keep_top_raw_text=True)
    _canonicalize_public_pipe_fields(cleaned)
    cleaned["raw_text"] = raw_text
    return cleaned


_CANONICAL_PIPE_FIELDS = {
    "origin_type",
    "formation_type",
    "primary_donor_language",
    "primary_donor_family",
}


def _canonicalize_public_pipe_fields(value: Any) -> Any:

    if isinstance(value, list):
        for item in value:
            _canonicalize_public_pipe_fields(item)
        return value
    if not isinstance(value, dict):
        return value

    for key, item in list(value.items()):
        if key in _CANONICAL_PIPE_FIELDS:
            value[key] = _canonicalize_pipe_value(item)
        else:
            _canonicalize_public_pipe_fields(item)
    return value


def _canonicalize_pipe_value(value: Any) -> Any:
    """Return a deduplicated representation for pipe-joined strings"""
    if not isinstance(value, str) or "|" not in value:
        return value
    parts = [part.strip() for part in value.split("|") if part.strip()]
    if len(parts) <= 1:
        return value
    return "|".join(dict.fromkeys(parts))


def _strip_redundant_raw_fields(value: Any, keep_top_raw_text: bool = False) -> Any:
    if isinstance(value, list):
        return [_strip_redundant_raw_fields(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned = {}
    for key, item in value.items():
        if key == "raw_text" and keep_top_raw_text:
            cleaned[key] = item
            continue
        if (key == "raw_text"
            or key == "text_raw"
            or key == "label_raw"
            or key == "summary_raw"
            or key.endswith("_raw")):
            continue
        if key in {
            "attestations_plain",
            "meanings_plain",
            "derivations_plain",
            "fossilized_inflected_forms_plain",
            "summary_normalized"}:
            continue
        out_key = _public_plain_key(key)
        cleaned[out_key] = _strip_redundant_raw_fields(item)
    return cleaned


def _public_plain_key(key: str) -> str:
    if key == "plain_text":
        return "text"
    if key.endswith("_parsed"):
        return key[:-len("_parsed")]
    if not key.endswith("_plain"):
        return key
    base = key[:-len("_plain")]
    if base in {"other_text", "other_etym_text"}:
        return base
    if base.endswith("_sum"):
        return base[:-len("_sum")] + "_summary"
    return base


def _collect_head_lines(lines: list[dict[str, str]]) -> list[str]:

    heads = []
    raw_lines = [ln["raw"] for ln in lines]

    for i, ln in enumerate(lines):
        raw = ln["raw"].strip()
        plain = ln["plain"].strip()

        # Stop at etymology / bibliography
        if ep.is_etymology_start_line(plain) or ep.is_book_start_line(plain):
            break

        if (ep.is_raw_bold_italic_head_line(raw)
            and (rx.HEAD_RE.match(plain)
                or ep.is_multiline_entry_start(raw_lines, i))):
            heads.append(raw)

    return heads

def _clean_head_line_raw(head_raw: str) -> str:

    if not head_raw:
        return ""

    return re.split(r'<b>\s*(?:\[\d+(?:/\d+)*\]\s*)?(?:A:|J:|Sz:|R:)',
        head_raw, maxsplit=1)[0].strip()


def _clean_entry_head_lines(entry: dict[str, Any]) -> None:
    for sub in entry.get("subentries", []):
        cleaned_raw = _clean_head_line_raw(sub.get("head_line_raw", ""))
        if not cleaned_raw:
            continue
        sub["head_line_raw"] = cleaned_raw
        sub["head_line_plain"] = nm.normalize_plain_text(cleaned_raw)

    entry["head_lines_raw"] = [sub.get("head_line_raw", "")
        for sub in entry.get("subentries", [])]
    entry["head_lines_plain"] = [sub.get("head_line_plain", "")
        for sub in entry.get("subentries", [])]


def _fill_subentries(entry: dict, lines: list[dict[str, str]]) -> None:

    subentries = entry.get("subentries", [])
    if not subentries:
        return

    head_positions = []
    for si, sub in enumerate(subentries):
        head_raw = sub["head_line_raw"]
        for li, ln in enumerate(lines):
            if ln["raw"].strip() == head_raw.strip():
                head_positions.append((si, li))
                break

    for pi, (si, start_li) in enumerate(head_positions):

        if pi + 1 < len(head_positions):
            end_li = head_positions[pi + 1][1]
        else:
            end_li = len(lines)

        for li in range(start_li + 1, end_li):
            plain = lines[li]["plain"].strip()
            if ep.is_etymology_start_line(plain) or ep.is_book_start_line(plain):
                end_li = li
                break

        sub = subentries[si]
        sub_lines = lines[start_li:end_li]

        _parse_subentry_zones(sub, sub_lines)


def _attach_family_etymologies(entry: dict, lines: list[dict[str, str]]) -> None:

    subentries = entry.get("subentries", [])
    if not subentries:
        return

    ety_chunks = []
    for chunk in bx.extract_etymology_chunks(lines):
        intro = chunk.get("intro_symbol")
        if intro == "■":
            ety_chunks.append(chunk)
        elif intro == "▣" and ety_chunks:
            ety_chunks[-1]["raw_text"] += "\n" + chunk["raw_text"]
            ety_chunks[-1]["plain_text"] += "\n" + chunk["plain_text"]

    if not ety_chunks:
        return

    if len(subentries) > 1 and len(ety_chunks) == 1:
        chunk = ety_chunks[0]
        entry["family_etymology"] = _parse_family_etymology_chunk(chunk)
        return

    chunk_targets = []
    for chunk in ety_chunks:
        explicit_targets = _explicit_targeted_subentry_indices(
            chunk.get("raw_text", ""),
            chunk.get("plain_text", ""),
            subentries)
        if explicit_targets:
            chunk_targets.append(explicit_targets)
            continue
        parsed = parse_etymology_with_analyses(chunk["raw_text"])
        target_text = " ".join([
            parsed.get("summary_raw", ""),
            parsed.get("summary_plain", "")])
        chunk_targets.append(_mentioned_subentry_indices(target_text, subentries))

    if any(chunk_targets):
        index_to_sub = {sub["subentry_index"]: sub for sub in subentries}
        assigned_indices = set()
        assigned_chunk_ids = set()
        for chunk, targets in zip(ety_chunks, chunk_targets):
            for index in targets:
                sub = index_to_sub.get(index)
                if sub:
                    etym = sub["subentry_etymology"]
                    if index in assigned_indices:

                        etym["etym_sum_raw"] = etym["etym_sum_raw"] + "\n" + chunk["raw_text"]
                        etym["etym_sum_plain"] = etym["etym_sum_plain"] + "\n" + chunk["plain_text"]
                        etym["blocks"] = bx.extract_symbolic_etymology_blocks(etym["etym_sum_raw"])
                    else:
                        _copy_etymology_chunk_to_subentry(sub, chunk)
                        assigned_indices.add(index)
                        assigned_chunk_ids.add(id(chunk))

        unassigned_subs = [sub for sub in subentries
            if sub.get("subentry_index") not in assigned_indices]
        untargeted_chunks = [chunk for chunk, targets in zip(ety_chunks, chunk_targets)
            if not targets and id(chunk) not in assigned_chunk_ids]
        if len(untargeted_chunks) == len(unassigned_subs) and untargeted_chunks:
            for sub, chunk in zip(unassigned_subs, untargeted_chunks):
                _copy_etymology_chunk_to_subentry(sub, chunk)
                assigned_indices.add(sub.get("subentry_index"))

        unassigned = [sub for sub in subentries
            if sub.get("subentry_index") not in assigned_indices]
        general_chunks = [chunk for chunk, targets in zip(ety_chunks, chunk_targets)
            if not targets and not _extract_bold_italic_lemmas(chunk.get("raw_text") or "")]
        if len(general_chunks) == 1 and unassigned:
            for sub in unassigned:
                _copy_etymology_chunk_to_subentry(sub, general_chunks[0])
                sub["_untargeted_family_etymology"] = True
        return

    if len(ety_chunks) == len(subentries):
        for sub, chunk in zip(subentries, ety_chunks):
            _copy_etymology_chunk_to_subentry(sub, chunk)
        return

    for sub, chunk in zip(subentries, ety_chunks):
        _copy_etymology_chunk_to_subentry(sub, chunk)

    if len(ety_chunks) > len(subentries):
        for chunk in ety_chunks[len(subentries):]:
            entry["unparsed_raw"].append(chunk["raw_text"])
            entry["unparsed_plain"].append(chunk["plain_text"])
        entry["parser_warnings"].append(
            "More etymology chunks than subentries; extra chunks left unparsed.")


_POS_SEPARATOR_RE = re.compile(r'\s*\|\s*(?=‹[^›]+›)')
_GLOSS_RE = re.compile(r"['\u2018\u2019]([^'\u2018\u2019]+)['\u2018\u2019]")


def _numbered_zone_items(raw_text: str, lemma: str | None = None) -> list[dict[str, Any]]:

    raw_text = raw_text.strip()
    if not raw_text:
        return []

    marker_re = re.compile(r'(?:<b>\s*)?(\d+)\.\s*</b>')
    matches = list(marker_re.finditer(raw_text))
    if matches:
        items = []
        for pos, match in enumerate(matches):
            start = match.start()
            end = matches[pos + 1].start() if pos + 1 < len(matches) else len(raw_text)
            item_raw = raw_text[start:end].strip()
            if re.match(r'^\d+\.\s*</b>', item_raw):
                item_raw = "<b>" + item_raw
            items.append(_parse_meaning_item(item_raw, int(match.group(1)), lemma))
        return items

    return [_parse_meaning_item(raw_text, None, lemma)]


def _parse_meaning_item(raw_text: str, index: int | None, lemma: str | None = None) -> dict[str, Any]:
    """Parse one numbered J: meaning into evidence/source snippets when possible"""
    plain = nm.normalize_plain_text(raw_text)
    plain = re.sub(r'^\d+\.\s*', '', plain)
    item: dict[str, Any] = {"index": index}

    compound_match = re.match(r'^\[([^\]~]+)~\]\s*', plain)
    if compound_match:
        prefix = compound_match.group(1).strip()
        item["form"] = _expand_tilde_with_lemma(f"{prefix}~", lemma) if lemma else f"{prefix}~"
        plain = plain[compound_match.end():]

    evidences = _meaning_evidence_items(plain, lemma)
    if evidences:
        item["attestations"] = evidences
    else:
        item["text_plain"] = plain
    return item


def _meaning_evidence_items(text: str, lemma: str | None = None) -> list[dict[str, Any]]:

    pos_parts = _POS_SEPARATOR_RE.split(text)
    if len(pos_parts) > 1:
        items = []
        for part in pos_parts:
            if part.strip():
                items.extend(_parse_pos_tagged_evidence(part.strip(), lemma))
        _resolve_same_as_previous_meanings(items)
        return items

    chunks = _split_evidence_chunks(text)
    items = []
    for chunk in chunks:
        evidence = _parse_evidence_chunk(chunk, parse_form=False, lemma=lemma)
        if evidence:
            items.append(evidence)
    _resolve_same_as_previous_meanings(items)
    return items


def _parse_pos_tagged_evidence(text: str, lemma: str | None = None) -> list[dict[str, Any]]:
    """Parse one POS-tagged meaning segment, e.g. '‹fn› 1573 'meaning | German' (source)'"""
    pos_match = re.match(r'^(‹[^›]+›)\s*', text)
    pos_tag = pos_match.group(1) if pos_match else None
    rest = text[pos_match.end():].strip() if pos_match else text
    chunks = _split_evidence_chunks(rest)
    items = []
    for chunk in chunks:
        evidence = _parse_evidence_chunk(chunk, parse_form=False, lemma=lemma)
        if evidence:
            if pos_tag:
                evidence["pos"] = pos_tag
            items.append(evidence)
    return items


def _resolve_same_as_previous_meanings(items: list[dict[str, Any]]) -> None:

    previous_hun: list[str] = []
    previous_ger: list[str] = []
    for item in items:
        hun = item.get("definition_hun", [])
        if hun == ["ua."] and previous_hun:
            item["definition_hun"] = previous_hun
            if previous_ger:
                item["translation_ger"] = previous_ger
            item["meaning_note"] = "same_as_previous"
        elif hun:
            previous_hun = hun
            previous_ger = item.get("translation_ger", [])


def _split_evidence_chunks(text: str) -> list[str]:

    chunks = []
    for part in _split_top_level_semicolon(text):
        chunks.extend(item.strip()
            for item in re.split(r'(?<=\))\s*,\s*(?=(?:\(?†?\d{3,4}|\d{1,2}\.\s*sz\.))', part)
            if item.strip())
    return chunks


def _parse_evidence_chunk(text: str, form: str | None = None,
    parse_form: bool = True, lemma: str | None = None) -> dict[str, Any] | None:
    """Parse one year/form/meaning/source chunk used inside meanings or derivations"""
    text = text.strip()
    if not text:
        return None

    source, source_start = _trailing_evidence_source(text)
    body = text[:source_start].strip() if source_start is not None else text
    meaning_match = _GLOSS_RE.search(body)
    meaning = meaning_match.group(0) if meaning_match else ""
    before_meaning = body[:meaning_match.start()].strip() if meaning_match else body
    year_match = re.search(r'\b(\d{3,4})\b', before_meaning)

    item: dict[str, Any] = {
        "year": int(year_match.group(1)) if year_match else None}
    if form:
        item["form"] = form
        date_text = _extract_evidence_date_text(before_meaning, year_match)
    elif parse_form:
        date_text, form_text, form_certainty = _split_evidence_date_and_form(before_meaning, year_match)
        if form_text:
            item["form"] = form_text
        if form_certainty:
            item["form_certainty"] = form_certainty
    else:
        date_text = before_meaning.strip()
    if date_text and not _evidence_date_text_is_redundant(date_text, item["year"]):
        item["date_text"] = date_text
    if meaning:
        # Strip surrounding quote marks and split Hungarian definition from German translation.
        inner = meaning[1:-1] if len(meaning) >= 2 else meaning
        if "|" in inner:
            hun, ger = inner.split("|", 1)
            item["definition_hun"] = _split_meaning_alternatives(hun)
            item["translation_ger"] = _split_meaning_alternatives(ger)
        else:
            item["definition_hun"] = _split_meaning_alternatives(inner)
    if source:
        item["source"] = source
    return item



""" SPLITTING DEFIITIONS INTO INDIVIDUAL WORDS IF POSSIBLE """
def _split_meaning_alternatives(text: str) -> list[str]:

    text = text.strip()
    if not text:
        return []

    semicolon_parts = [s.strip() for s in text.split(";") if s.strip()]
    if len(semicolon_parts) > 1:
        parts: list[str] = []
        for part in semicolon_parts:
            parts.extend(_split_meaning_alternatives(part))
        return parts

    comma_parts = [s.strip() for s in text.split(",") if s.strip()]
    if len(comma_parts) == 2 and all(_is_single_word_gloss_item(part) for part in comma_parts):
        return comma_parts
    if len(comma_parts) >= 3 and all(_is_single_word_gloss_item(part) for part in comma_parts):
        return comma_parts

    return [text]


def _is_single_word_gloss_item(text: str) -> bool:

    value = text.strip()
    if not value:
        return False
    value = re.sub(r"\s*[〈‹<][^〉›>]+[〉›>]\s*", "", value).strip()
    words = [part for part in value.split() if part]
    return len(words) == 1


def _split_evidence_date_and_form(before_meaning: str,
    year_match: re.Match[str] | None) -> tuple[str, str, str | None]:

    if not year_match:
        return before_meaning.strip(), "", None

    date_text = before_meaning[:year_match.end()].strip()
    rest = before_meaning[year_match.end():].strip()
    form_certainty = None
    marker_parts: list[str] = []

    marker_re = re.compile(
        r'^(?:'
        r'u\./?[¹²³⁴⁵⁶⁷⁸⁹⁰\d]*|'
        r'k\.|'
        r'/[¹²³⁴⁵⁶⁷⁸⁹⁰\d]*|'
        r'\?'
        r')\s*',
        flags=re.IGNORECASE,)
    while rest:
        marker_match = marker_re.match(rest)
        if not marker_match:
            break
        marker = marker_match.group(0).strip()
        if marker:
            marker_parts.append(marker)
            if "?" in marker:
                form_certainty = "uncertain"
        rest = rest[marker_match.end():].strip()

    if marker_parts:
        for marker in marker_parts:
            if marker.startswith("/"):
                date_text = f"{date_text}{marker}"
            else:
                date_text = f"{date_text} {marker}"

    form_text = re.sub(r'\s+\[[^\]]+\]', '', rest).strip()
    return date_text, form_text, form_certainty


def _evidence_date_text_is_redundant(date_text: str, year: int | None) -> bool:

    if year is None:
        return False
    remainder = date_text.replace(str(year), "", 1).strip()
    remainder = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+', '', remainder)
    remainder = re.sub(r'\bu\./?', '', remainder, flags=re.IGNORECASE)
    remainder = re.sub(r'[/\?\s]+', '', remainder)
    return not remainder


def _trailing_evidence_source(raw_text: str) -> tuple[str, int | None]:

    source, source_start = _trailing_parenthesized_source(raw_text)
    if source != "(↑)" or source_start is None:
        return source, source_start

    stripped = raw_text[:source_start].rstrip()
    previous_source, previous_start = _trailing_parenthesized_source(stripped)
    if previous_source and previous_start is not None:
        return previous_source, previous_start
    return "", None


def _extract_evidence_date_text(before_meaning: str, year_match: re.Match[str] | None) -> str:
    """Return a non-year date phrase such as '12. sz. vége/', if present"""
    if not year_match:
        return before_meaning.strip()
    return before_meaning[:year_match.end()].strip()



def _split_top_level_semicolon(raw_text: str) -> list[str]:

    parts = []
    start = 0
    depth = 0
    in_quote = False
    for index, char in enumerate(raw_text):
        if char in {"'", "\u2018", "\u2019"}:
            in_quote = not in_quote
        elif char in "([" and not in_quote:
            depth += 1
        elif char in ")]" and depth and not in_quote:
            depth -= 1
        elif char == ";" and depth == 0 and not in_quote:
            parts.append(raw_text[start:index + 1].strip())
            start = index + 1
    tail = raw_text[start:].strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


_CENTURY_DATE_TERMS = [
    "eleje",
    "végén",
    "vége",
    "közepe",
    "közepén",
    "első\\s+fele",
    "második\\s+fele",
    "első\\s+harmada",
    "második\\s+harmada",
    "utolsó\\s+harmada",
    "e\\."]

_CENTURY_DATE_RE = re.compile(
    r"(?<!\d)"
    r"(?:[\[\(†]\s*)?"
    r"(?P<century>\d{1,2})"
    r"(?:\s*[–-]\s*(?P<century_end>\d{1,2}))?"
    r"\.\s*(?:sz\.|század\b)"
    rf"(?P<modifier>(?:\s*(?:{'|'.join(_CENTURY_DATE_TERMS)}))*)"
    r"\s*(?:\])?\s*(?:\?)?\s*(?:[/?])?",
    re.IGNORECASE)


def _century_year_range(century: int, modifier: str, century_end: int | None = None) -> tuple[int, int]:

    start_base = (century - 1) * 100
    end_base = ((century_end or century) - 1) * 100
    modifier = (modifier or "").casefold()

    if century_end is not None:
        return start_base + 1, end_base + 99
    if "vége" in modifier or "végén" in modifier:
        return start_base + 80, start_base + 99
    if "közepe" in modifier or "közepén" in modifier:
        return start_base + 40, start_base + 60
    if re.search(r"\bmásodik\s+fele\b", modifier):
        return start_base + 51, start_base + 99
    if re.search(r"\belső\s+fele\b", modifier):
        return start_base + 1, start_base + 50
    if re.search(r"\belső\s+harmada\b", modifier):
        return start_base + 1, start_base + 33
    if re.search(r"\bmásodik\s+harmada\b", modifier):
        return start_base + 34, start_base + 66
    if re.search(r"\butolsó\s+harmada\b", modifier):
        return start_base + 67, start_base + 99
    if "eleje" in modifier or re.search(r"\be\.", modifier):
        return start_base + 1, start_base + 20
    return start_base + 1, start_base + 99


def _deterministic_year_in_range(start: int, end: int, seed_text: str) -> int:
    """Choose a stable pseudo-random year inside an inclusive range"""
    if end <= start:
        return start
    digest = hashlib.sha256(seed_text.encode("utf-8", errors="ignore")).digest()
    offset = int.from_bytes(digest[:8], "big") % (end - start + 1)
    return start + offset


def _generated_century_year_info(date_zone_raw: str, raw_text: str, index: int) -> dict[str, Any] | None:
    """Return generated-year metadata for century-style dates such as '12. sz. vége/'"""
    date_zone_plain = nm.normalize_plain_text(date_zone_raw)
    match = _CENTURY_DATE_RE.search(date_zone_plain)
    if not match:
        return None

    century = int(match.group("century"))
    century_end_text = match.group("century_end")
    century_end = int(century_end_text) if century_end_text else None
    start, end = _century_year_range(century, match.group("modifier") or "", century_end)
    source_text = match.group(0).strip()
    generated_year = _deterministic_year_in_range(
        start,
        end,
        f"{index}|{source_text}|{nm.normalize_plain_text(raw_text)}",
    )
    return {"year": generated_year,
        "year_is_generated": True,
        "year_generated_from": source_text,
        "year_range": [start, end],
        "year_generation_method": "deterministic_random_within_century_range"}


def _parse_attestation_item(raw_text: str, index: int) -> dict[str, Any]:
    """Parse one semicolon-separated A: attestation into compact public fields"""
    plain = nm.normalize_plain_text(raw_text)
    source_raw, source_start = _trailing_parenthesized_source(raw_text)
    form_zone_raw = raw_text[:source_start] if source_start is not None else raw_text
    first_form_start = form_zone_raw.find("<i>")
    date_zone_raw = form_zone_raw[:first_form_start] if first_form_start >= 0 else form_zone_raw
    year_match = re.search(r'\b(\d{3,4})\b', nm.normalize_plain_text(date_zone_raw))
    forms_raw = [form.strip() for form in re.findall(r'<i>(.*?)</i>', form_zone_raw, re.DOTALL) if form.strip()]
    forms_plain = [nm.normalize_plain_text(form) for form in forms_raw]
    date_type = _attestation_date_type(raw_text, form_zone_raw, year_match.group(1) if year_match else None)
    generated_year_info = _generated_century_year_info(date_zone_raw, raw_text, index)
    use_generated_year = generated_year_info is not None and (year_match is None or date_type == "excluded_antedated")
    form_certainty = _attestation_form_certainty(form_zone_raw, year_match)
    item = {"text_raw": raw_text,
        "year": (generated_year_info["year"]
            if use_generated_year
            else int(year_match.group(1)) if year_match else None),
        "form": forms_plain[0] if forms_plain else "",
        "source": nm.normalize_plain_text(source_raw)}
    if use_generated_year:
        item.update(generated_year_info)
        item["date_type"] = ("generated_century_from_antedated"
            if date_type == "excluded_antedated"
            else "generated_century")
        if year_match and date_type == "excluded_antedated":
            item["excluded_antedated_year"] = int(year_match.group(1))
    elif date_type:
        item["date_type"] = date_type
    if form_certainty:
        item["form_certainty"] = form_certainty
    if len(forms_plain) > 1:
        item["forms"] = forms_plain
    return item


def _attestation_date_type(raw_text: str, form_zone_raw: str, year: str | None) -> str | None:

    if not year:
        return None
    year_match = re.search(re.escape(year), raw_text)
    if not year_match:
        return None
    prefix = raw_text[max(0, year_match.start() - 4):year_match.start()]
    if "†" in prefix:
        return "excluded_antedated"

    form_start = form_zone_raw.find("<i>")
    prefix_before_year = raw_text[:year_match.start()]
    between_year_and_form = form_zone_raw[year_match.end():form_start if form_start >= 0 else len(form_zone_raw)]
    if "/" in prefix_before_year or "/" in between_year_and_form:
        return "copy_or_later_print"
    if "?" in prefix_before_year:
        return "uncertain"
    return None


def _attestation_form_certainty(form_zone_raw: str, year_match: re.Match[str] | None) -> str | None:
    """Mark uncertain spellings when '?' appears after the date but before the form"""
    if not year_match:
        return None
    form_start = form_zone_raw.find("<i>")
    if form_start < 0:
        return None
    between_year_and_form = form_zone_raw[year_match.end():form_start]
    return "uncertain" if "?" in between_year_and_form else None


def _trailing_parenthesized_source(raw_text: str) -> tuple[str, int | None]:
    """Return the final balanced parenthesized source and its start offset"""
    text = raw_text.rstrip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    if not text.endswith(")"):
        return "", None

    depth = 0
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                return text[index:], index
    return "", None


def _attestation_zone_items(raw_text: str) -> list[dict[str, Any]]:
    """Split and parse the A: attestation zone"""
    return [_parse_attestation_item(part, index)
        for index, part in enumerate(_split_top_level_semicolon(raw_text.strip()), start=1)]


def _lengthen_final_vowel(word: str) -> str:
    """Lengthen the final Hungarian vowel for derivations marked by accented tilde.
    for example, 'haza' -> ~´m -> 'hazám' instead of the incorrect 'hazam' """
    if not word:
        return word
    length_map = str.maketrans({
        "a": "á",
        "e": "é",
        "i": "í",
        "o": "ó",
        "ö": "ő",
        "u": "ú",
        "ü": "ű",
        "A": "Á",
        "E": "É",
        "I": "Í",
        "O": "Ó",
        "Ö": "Ő",
        "U": "Ú",
        "Ü": "Ű",
    })
    return word[:-1] + word[-1].translate(length_map)


def _expand_tilde_with_lemma(text: str, lemma: str | None) -> str:

    if not lemma or "~" not in text:
        return text
    # Strip trailing -ik for ik-verbs so ~ó -> folyó, not folyikó
    stem = lemma[:-2] if lemma.endswith("ik") else lemma
    long_stem = _lengthen_final_vowel(stem)
    text = re.sub(r"[´′`ˊ]\s*~", long_stem, text)
    text = re.sub(r"~\s*[́´′`ˊ]", long_stem, text)
    return text.replace("~", stem)


def _bold_pipe_zone_items(raw_text: str, lemma: str | None = None) -> list[dict[str, Any]]:

    raw_text = raw_text.strip()
    if not raw_text:
        return []

    parts = re.split(r'\s+\|\s+(?=<b>)', raw_text)
    items = []
    for index, part in enumerate(parts, start=1):
        part = part.strip()
        if not part:
            continue
        plain = nm.normalize_plain_text(part)
        items.append(_parse_derivation_item(part, index, lemma))
    return items


def _parse_derivation_item(raw_text: str, index: int, lemma: str | None) -> dict[str, Any]:
    """Parse one Sz:/R: item into a derived form and its evidence snippets"""
    plain = _expand_tilde_with_lemma(nm.normalize_plain_text(raw_text), lemma)
    # match 3-4 digit years AND century references like "13. sz."
    first_year = re.search(r'\b\d{1,2}\.\s*sz\b|\b\d{3,4}\b', plain)
    item: dict[str, Any] = {"index": index}
    if not first_year:
        item["text_plain"] = plain
        return item

    head = plain[:first_year.start()].strip()
    forms = _derivation_head_forms(head)
    item["form"] = forms  # always a list
    note = _derivation_head_note(head)
    if note:
        item["note"] = note

    evidence_text = plain[first_year.start():].strip()
    evidences = []
    for chunk in _split_evidence_chunks(evidence_text):
        evidence = _parse_evidence_chunk(chunk)
        if evidence:
            evidences.append(evidence)
    if evidences:
        item["attestations"] = evidences
    else:
        item["text_plain"] = plain
    return item


def _clean_derivation_form(form: str) -> str:
    """Strip trailing dagger/obsolete markers from a derivation form"""
    # remove "( †)", "(†", "(†)", bare "†" and everything after
    form = re.sub(r'\s*\(\s*[†âœ][^)]*\)?\s*$', '', form).strip()
    form = re.sub(r'\s*[†âœ].*$', '', form).strip()
    return form


def _expand_optional_parts(form: str) -> list[str]:

    form = _clean_derivation_form(form)
    if not form:
        return []

    # optional prefix at the start: (el)korcsosul
    prefix_match = re.match(r'^\(([^)†]+)\)(.+)$', form)
    if prefix_match:
        opt, rest = prefix_match.groups()
        return [opt + rest, rest]

    # optional infix or suffix: figuráz(ik), hallgató(d)zik
    suffix_match = re.match(r'^([^(]+)\(([^)†]+)\)(.*)$', form)
    if suffix_match:
        before, opt, after = suffix_match.groups()
        return [before + after, before + opt + after]

    return [form]


def _derivation_head_forms(head: str) -> list[str]:
    """Return all derived forms before any bracketed note, fully expanded"""
    before_bracket = re.split(r'\s+\[', head, maxsplit=1)[0].strip()
    raw_forms = [f.strip() for f in before_bracket.split(",") if f.strip()]
    if not raw_forms:
        raw_forms = [before_bracket]
    expanded = []
    for f in raw_forms:
        expanded.extend(_expand_optional_parts(f))
    return expanded if expanded else [before_bracket]



def _derivation_head_note(head: str) -> str:
    """Return a bracketed note after the derived form, if there is one"""
    match = re.search(r'(\[[^\]]+\])', head)
    return match.group(1) if match else ""


def _parse_subentry_zones(sub: dict, sub_lines: list[dict[str, str]]) -> None:

    if not sub_lines:
        return

    head_plain = sub["head_line_plain"]

    #  Parse lemma / homograph / flags from head 
    m = rx.HEAD_RE.match(head_plain)
    if m:
        sub["lemma"] = m.group(1)
        sup = m.group(2)
        sub["homograph"] = int(sup.translate(rx.SUPERSCRIPT_MAP)) if sup else None
        stemtype = m.group(3)
        sub["stemtype"] = stemtype if stemtype else None
    else:
        raw_head_match = rx.RAW_BOLD_ITALIC_HEAD_RE.match(sub.get("head_line_raw", ""))
        if raw_head_match:
            sub["lemma"] = raw_head_match.group(1)
            sup = raw_head_match.group(2)
            sub["homograph"] = int(sup.translate(rx.SUPERSCRIPT_MAP)) if sup else None

    pre_a_plain = []
    for ln in sub_lines:
        plain = ln["plain"]
        if "A:" in plain:
            break
        pre_a_plain.append(plain)
    head_before_a = re.split(r'\bA:', head_plain, maxsplit=1)[0]
    head_zone_plain = " ".join([head_before_a, *pre_a_plain])

    if "†" in head_zone_plain:
        sub["flags"]["is_obsolete"] = True
    if "△" in head_zone_plain or "∆" in head_zone_plain:
        sub["flags"]["is_archaic"] = True
    if "×" in head_zone_plain:
        sub["flags"]["is_dialectal"] = True

    #  Collect zone texts 
    full_raw = "\n".join(ln["raw"] for ln in sub_lines)
    full_plain = "\n".join(ln["plain"] for ln in sub_lines)

    section_tag = {"A": r'<b>\s*(?:\[\d+(?:/\d+)*\]\s*)?A:[^<]*</b>\s*',
        "J": r'(?:<b>\s*J:\s*</b>\s*|<b>\s*J:\s*)',
        "Sz": r'<b>\s*Sz:[^<]*</b>\s*',
        "R": r'<b>\s*R:[^<]*</b>\s*'}

    # A: block, between "A:" and "J:" (or Sz: or ■)
    a_match = re.search(section_tag["A"] + rf'(.*?)(?={section_tag["J"]}|{section_tag["Sz"]}|{section_tag["R"]}|■|$)',
        full_raw, re.DOTALL)
    if a_match:
        sub["attestations_raw"] = a_match.group(1).strip()
        sub["attestations_plain"] = nm.normalize_plain_text(a_match.group(1))
        sub["attestations_parsed"] = _attestation_zone_items(a_match.group(1))

    # J: block -> meaning and translation
    j_match = re.search(section_tag["J"] + rf'(.*?)(?={section_tag["Sz"]}|{section_tag["R"]}|■|$)',
        full_raw, re.DOTALL)
    if j_match:
        sub["meanings_raw"] = j_match.group(1).strip()
        sub["meanings_plain"] = nm.normalize_plain_text(j_match.group(1))
        sub["meanings_parsed"] = _numbered_zone_items(j_match.group(1), sub.get("lemma"))

    # Sz: block -> derivations
    sz_match = re.search(section_tag["Sz"] + rf'(.*?)(?={section_tag["R"]}|■|$)',
        full_raw, re.DOTALL)
    if sz_match:
        sub["derivations_raw"] = sz_match.group(1).strip()
        sub["derivations_plain"] = nm.normalize_plain_text(sz_match.group(1))
        sub["derivations_parsed"] = _bold_pipe_zone_items(sz_match.group(1), sub.get("lemma"))

    # R: block -> fossilized forms
    r_match = re.search(section_tag["R"] + r'(.*?)(?=■|$)',
        full_raw, re.DOTALL)
    if r_match:
        sub["fossilized_inflected_forms_raw"] = r_match.group(1).strip()
        sub["fossilized_inflected_forms_plain"] = nm.normalize_plain_text(r_match.group(1))
        sub["fossilized_inflected_forms_parsed"] = _bold_pipe_zone_items(r_match.group(1), sub.get("lemma"))

    #  Etymology chunk (■ ... up to 🕮) 
    ety_match = re.search(r'(■\s+.*?)(?=🕮|$)', full_raw, re.DOTALL)
    if ety_match:
        etym_chunk = ety_match.group(1).strip()
        sub["subentry_etymology"]["etym_sum_raw"] = etym_chunk
        sub["subentry_etymology"]["etym_sum_plain"] = nm.normalize_plain_text(etym_chunk)


def _fill_book_block(entry: dict, lines: list[dict[str, str]]) -> None:
    """Extract 🕮 bibliography lines"""
    book_raw = []
    book_plain = []
    in_book = False

    for ln in lines:
        plain = ln["plain"].strip()
        if plain.startswith("🕮"):
            in_book = True
        if in_book:
            book_raw.append(ln["raw"].strip())
            book_plain.append(plain)

    if book_raw:
        raw_text = " ".join(book_raw)
        plain_text = " ".join(book_plain)
        book_plain_text, see_also_plain = _split_book_and_see_also(plain_text)
        book_raw_text, see_also_raw = _split_book_and_see_also(raw_text)
        entry["book_block_raw"] = [book_raw_text]
        entry["book_block_plain"] = [book_plain_text]
        if see_also_plain:
            entry["see_also_plain"] = _split_see_also_items(see_also_plain)
            entry["see_also_raw"] = _split_see_also_items(see_also_raw)


def _split_book_and_see_also(text: str) -> tuple[str, str]:
    """Separate bibliography text from the trailing 'Vö.' cross-reference list"""
    text = text.strip()
    match = re.search(r'\s*•\s*Vö\.\s*', text)
    if not match:
        return text, ""
    return text[:match.start()].strip(), text[match.end():].strip()


def _split_see_also_items(text: str) -> list[str]:
    """Split a comma-separated 'Vö.' list into cleaned cross-reference strings"""
    text = text.strip().rstrip(".")
    if not text:
        return []
    return [nm.normalize_plain_text(item).strip()
        for item in text.split(",")
        if nm.normalize_plain_text(item).strip()]


def _determine_family_base(entry: dict) -> list[int]:

    subentries = entry.get("subentries", [])
    if not subentries:
        return [0]

    family_etym = entry.get("family_etymology") or {}
    if (family_etym.get("origin_type")
        and not family_etym.get("origin_hypotheses")
        and not family_etym.get("analyses")):
        return _earliest_subentry_indices(subentries)

    lemma_to_index = {sub.get("lemma"): sub.get("subentry_index")
        for sub in subentries if sub.get("lemma") and sub.get("subentry_index") is not None}
    explicit_bases = []
    earlier_family_heads = []
    for analysis in family_etym.get("analyses") or []:
        label = (analysis.get("label_plain") or "").casefold()
        targets = analysis.get("target_lemmas") or []
        if "szócsalád alapja" in label or "szócsalád korábbi tagja" in label:
            explicit_bases.extend(lemma_to_index.get(lemma) for lemma in targets)
        elif "szócsalád legkorábbi tagja" in label:
            earlier_family_heads.extend(lemma_to_index.get(lemma) for lemma in targets)

    explicit_bases = [idx for idx in dict.fromkeys(explicit_bases) if idx is not None]
    if explicit_bases:
        return explicit_bases

    earlier_family_heads = [idx for idx in dict.fromkeys(earlier_family_heads) if idx is not None]
    if earlier_family_heads:
        return earlier_family_heads

    explicit_subentry_bases = []
    for sub in subentries:
        summary = nm.normalize_plain_text((sub.get("subentry_etymology") or {}).get("summary_plain") or "").casefold()
        if any(phrase in summary for phrase in (
            "szócsalád alapja",
            "szócsalád legkorábbi tagja",
            "szócsalád korábbi tagja")):
            explicit_subentry_bases.append(sub.get("subentry_index"))

    explicit_subentry_bases = [
        idx for idx in dict.fromkeys(explicit_subentry_bases) if idx is not None]
    if explicit_subentry_bases:
        return explicit_subentry_bases

    bases = []
    for sub in subentries:
        etym = sub.get("subentry_etymology", {})
        summary = (etym.get("summary_plain") or "").lower()
        normalized = etym.get("summary_normalized") or ""

        # If it's described as a derivative/back-formation of another word in this family,
        # it's NOT the base
        derived_tags = {
            "<DERIVATION>",
            "<CONVERSION>",
            "<BACK_FORMATION>",
            "<REDUPLICATION>",
            "<FOSSILIZATION>",
            "<CONSTITUENT_LEXICALIZATION>",
            "<NYELVUJITAS>"}
        is_derived = any(tag in normalized for tag in derived_tags) or any(kw in summary for kw in [
            "származékszó",
            "szófajváltás",
            "elvonás",
            "kicsinyítő képzős származéka",
            "analógiás alakulat"])
        if not is_derived:
            bases.append(sub["subentry_index"])

    return bases if bases else [subentries[0]["subentry_index"]]