"""
Normalize short Hungarian etymology summaries into broad tags.
The parser uses these tags as an intermediate signal for origin type,
formation type, language family, and disputedness. The functions here do not
decide the full JSON structure. they make summary strings more regular so the
main parser can assign metadata without repeating thousands of regexes.
"""

from . import ueszweb_abbreviations as ab
import re
from typing import Any, Dict

NORMALIZATION_PATTERNS = {
    
    "<INHERITED>": [
        r"\balapnyelvi örökség\b",
        r"\balapnyelvi\b",
        (
            r"\börökség\b"
            r"(?:\s*,?\s*"
            r"(?:finnugor|ugor|uráli)"
            r"(?:\s+kori)?"
            r"(?:\s+tővel|\s+korból)?"
            r")*"
        ),
    ],
    
    # ORIGIN_TYPE
    "<LOAN_WORD>": [
        r"\bjövevényszó\b",
        r"\bjövevényszók\b",
        r"\bjövevényszavak\b",
        r"\bjövevényszavai\b",
        r"jövevényszó",
        r"\b(?:francia|olasz|német|angol|latin|görög|török|szláv|orosz|spanyol|portugál)"
        r"\s+tulajdonnév(?:i|ből|ből\s+származó)?\b",
        r"\b(?:francia|olasz|német|angol|latin|görög|török|szláv|orosz|spanyol|portugál)"
        r"\s+tulajdonnévi\s+köznevesülés\b",
    ],
    
    # ORIGIN_TYPE
    "<INTERNATIONAL>" : [
        r"\bnemzetközi szó\b",
        r"\bnemzetközi szók\b",
        r"\bnemzetközi szavak\b",
        r"\bnemzetközi\b",
    ],
    
     # ORIGIN_TYPE
    "<WANDERWORD>" : [
        r"\bvándorszók\b",
        r"\bvándorszó\b",
    ],   

    # KÖZVETITVE ... NYELV ALTAL
    "<INTERMEDIARY>": [
        r"\bközvetítéssel\b",
        r"\bközvetítéssel is\b",
    ],

    # ORIGIN_TYPE
    "<ONOMATOPOETIC>": [
        r"\bonomatopoetikus szó\b",
        r"\bonomatopoetikus szavak\b",
        r"\bonomatopoetikus eredetűek\b",
        r"\bonomatopoetikus eredetű\b",
        r"\bonomatopoetikus eredetű szó\b",
        r"\bonomatopoetikus eredetű szavak\b",
    ],

    # ORIGIN_TYPE
    "<FICTIONAL_PLAYFUL>": [
        r"\brelatív fiktív tő származékából\b",
        r"\brelatív fiktív tőből\b",
        r"\bfiktív tőből\b",
        r"\bfiktív tő származékából\b",
        r"\bfiktív tő származékának\b",
        r"\brelatív fiktív tőből\b",
        r"\brelatív fiktív tő\b",
        r"\bfiktív tőből\b",
        r"\bfiktív tövű\b",
        r"\bjátszi szóalkotás útján\b",
        r"\bjátszi szóalkotással\b",
        r"\bjátszi szóalkotás\b",
    ],
    
    # BELSÖ KELETKEZESU _ ORIGIN_TYPE
    "<INTERNAL_ORIGIN>": [
        r"\bbelső fejlődés\b",
        r"\bbelső fejlemény\b",
        r"\bbelső keletkezésűek\b",
        r"\bbelső keletkezésű\b",
        r"\bbelső fejlődés eredménye\b",
        r"\bmagyar fejlemény\b",  
        r"\bmagyar nyelvi fejlemény\b",
        r"\bmagyar tulajdonnévből\b",    
        r"\bköznevesülés egy magyar\b",   
        r"\bköznévvé vált magyar tulajdonnév\b", 
        r"\bmagyar szószerkezet\b",            
    ],
    
    # SZOFAJVALTASSAL _ FORMATION_TYPE
    "<CONVERSION>" : [
        r"\bszófajváltással\b",
        r"\bszófajváltásának eredménye\b",
        r"\bszófajváltás eredménye\b",
        r"\bszófajváltása\b",
        r"\bszófajváltás\b",
        r"\bszófajváltás általi\b",
    ],
    
    # TUDATOS SZOALKOTASSAL / ORIGIN_TYPE
    "<COINAGE>": [
        r"\btudatos szóalkotás útján\b",
        r"\btudatos szóalkotással\b",
        r"\btudatos szóalkotás\b",
        r"\btudatos\b",
    ],
    
    # REDUPLICATIONS - IKERSZÓ
    "<REDUPLICATION>": [
        r"\bikerszók\b",
        r"\bikerszó\b",
    ],    
    
    # MEGSZILARDULT RAGOS ALAKULAT
    "<FOSSILIZATION>": [
        r"\bmegszilárdult ragos alakulatok\b",
        r"\bmegszilárdult ragos alakulat\b",
        r"\bmegszilárdult ragos\b",
        r"\bragszilárdulás\b",
    ],
    
    # SZOÖSSZETETEL --> 
    "<PHRASAL_COMPOUND>": [
        r"\bszóösszetétel\b",
        r"\bösszetétel\b(?!\s+(?:előtag|utótag))",
    ],
    
    "<WORD_FAMILY>" : [
        r"\bszócsalád\b",
        r"\bszócsalád alapja\b",
        r"\bszócsalád alapjai\b",
        ],
    
    # SEMANTIC/LEXICAL SPLIT
    "<LEXICAL_SPLIT>" : [
        r"\bszóhasadás eredménye\b",
        r"\bszóhasadás\b",
        r"\bszóhasadással\b",
        ],
    
    # tisztazatlan viszony
    "<UNCLEAR_RELATION>": [
        r"\btisztázatlan viszonyú tagokkal\b",
        r"\btagok egymáshoz való viszonya tisztázatlan\b",
        r"\bmelynek tagjai tisztázatlan viszonyban állnak egymással\b",
        r"\btisztázatlan viszonyú\b",
        r"\btagjainak viszonya tisztázatlan\b",
        r"\bamelynek tagjai tisztázatlan viszonyban állnak egymással\b",
        r"\btagjainak egymáshoz való viszonya bizonytalan\b",
        r"\btagok egymáshoz való viszonya bizonytalan\b",
    ],
    
    # FORMATION_TYPE
    "<DERIVATION>": [
        r"\bszármazékszavak\b",
        r"\bszármazékszók\b",
        r"\bszármazékszó\b",
    ],
    
    # HANGRENDI ÁTCSAPÁS
    "<VOWEL_HARMONY_SHIFT>" : [
        r"\bHangrendi átcsapással keletkezett\b",
        r"\bhangrendi átcsapás\b",
        
        ],

    # FORMATION_TYPE
    "<COMPOUND>": [
        r"\bösszetett szók\b",
        r"\bösszetett szó\b",
        r"\bösszetett szavak\b",
    ],

    # ORIGIN TYPE
    "<CALQUE>": [
        r"\bmintára alkotott tükörfordítás\b",
        r"\bmintára tükörszó\b",
        r"\bképzett tükörszó\b",
        r"\btükörfordítások\b",
        r"\btükörfordítás\b",
    ],
    
    "<LOANBLEND>" : [
        r"\bmintára alkotott részfordítás\b",
        r"\bmintájú részfordítás\b",
        r"\brészfordítás\b",
        ],
    
    # FORMATION_TYPE
    "<BACK_FORMATION>" : [
        r"\belvonással\b",
        r"\belvonás eredménye\b",
        r"\belvonás\b",
        r"\belvonások\b",
        ],
    
    # ELÖTAG-UTOTAG --> FORMATION_TYPE
    "<CONSTITUENT_LEXICALIZATION>" : [
        r"\bösszetett szó (előtagjának|utótagjának) (önállósulása|önállósodása)\b",
        r"\bösszetételek (előtagjának|utótagjának) (önállósulása|önállósodása)\b",
        r"\bösszetétel (előtagjának|utótagjának) (önállósulása|önállósodása)\b",
        r"\bösszetételi (előtag|utótag) önállósodásával jött létre\b",
        r"\bösszetételi (előtag|utótag) (önállósulása|önállósodása)\b",
        r"\bösszetételek (előtagja|utótagja)\b",
        r"\bösszetett szavak (előtagja|utótagja)\b",
        r"\bösszetétel (előtagja|utótagja)\b",
        r"\bösszetételi (előtag|utótag)\b",
        r"\btükörfordítások (előtagja|utótagja)\b",
        r"\btükörszavak (előtagja|utótagja)\b",
        r"\bösszetett szavak (előtagja|utótagja)\b",
        r"\brészfordítás (előtagja|utótagja)\b",
        r"\btükörfordítás (előtagja|utótagja)\b",
        r"\bösszetett szók (előtagja|utótagja)\b",
        r"\brészfordítások (előtagja|utótagja)\b",
        r"\bösszetételek (előtagjának|utótagjának) önállósulásával\b",
        r"\bösszetétel (előtagjának|utótagjának) önállósulásával\b",
        r"(?:<compound>|<calque>)\s+(?:előtagja|utótagja)i?",
        ],


    "<UNKNOWN_ORIGIN>": [
        r"\bismeretlen eredetű\b",
        r"\betimológiája ismeretlen\b",
    ],
    
    
    "<UNCERTAIN_BASE>": [
        r"\balapszava bizonytalan\b",
        r"\balapszó bizonytalan\b",
        r"\bmegfelelői bizonytalanok\b",
    ],    
    
    
    "<UNCERTAIN_ORIGIN>": [
        r"\bátadó nyelv tisztázatlan\b",
        r"\bbizonytalan eredetűek\b",
        r"\bbizonytalan eredetű\b",
        r"\bátadó nyelv bizonytalan\b",
    ],

    "<UNCERTAIN_FORMATION>": [
        r"\bkeletkezési módja bizonytalan\b",
        r"\bkeletkezésmódja bizonytalan\b",
        r"\bkialakulásmódja bizonytalan\b",
        r"\bképzésmódja bizonytalan\b",
        r"\bkeletkezésmódjuk bizonytalan\b",
        r"\balakulásmódja bizonytalan\b",
        r"\bképzési módja bizonytalan\b",
    ],

    # VITATOTT EREDET
    "<DISPUTED_ORIGIN>": [
        r"\bvitatott eredetű\b",
        r"\bátadó nyelv vitatott\b",
        r"\bvitatott eredetűek\b",

    ],


    "<DISPUTED_FORMATION>": [
        r"\bkeletkezésmódja vitatott\b",
        r"\bkialakulásmódja vitatott\b",
    ],

    "<DISPUTED_COGNATES>": [
        r"\bmegfelelések vitatottak\b",
        r"\bmegfelelői vitatottak\b",
        r"\begyeztetés vitatott\b",
    ],
    
    "<DISPUTED_BASE>" : [
        r"\balapszó vitatott\b",
        r"\balapszava vitatott\b",
        ],
    

    "<NYELVUJITAS>": [
        r"\bnyelvújítási szó\b",
        r"\bnyelvújítás korában\b",
    ],

    "<PROBABILITY>": [
        r"\btalán\b",
        r"\besetleg\b",
        r"\bvalószínűleg\b",
        r"\bfeltehetőleg\b",
        r"\blehet\b",
    ],
}

def build_lang_to_family(language_families):
    lang_to_family = {}
    
    for family, langs in language_families.items():
        for lang in langs:
            lang_to_family[lang.casefold()] = family
            
    return lang_to_family


LANG_TO_FAMILY = build_lang_to_family(ab.LANGUAGE_FAMILIES)

cardinal_slavic_variants_in_summaries = [
    "déli szláv",
    "keleti szláv",
    "nyugati szláv"
]

csuvas_turkic_summaries = ["csuvasos típusú ótörök nyelvből", "oszmán-török"]

donor_lang = ["átadó nyelv"]

def apply_normalization_patterns(text: str, patterns: dict[str, list[str]]) -> str:

    all_patterns = []

    for label, regex_list in patterns.items():
        for regex in regex_list:
            regex = regex.strip()
            if not regex or regex == r"\b\b":
                continue
            all_patterns.append((regex, label))

    # longest / most specific first
    all_patterns.sort(key=lambda x: len(x[0]), reverse=True)

    # split into tags and non-tags
    parts = re.split(r"(<[^>]+>)", text)

    for i, part in enumerate(parts):
        # skip already inserted tags
        if re.fullmatch(r"<[^>]+>", part):
            continue

        for pattern, label in all_patterns:
            part = re.sub(pattern, label, part, flags=re.IGNORECASE)

        parts[i] = part

    return "".join(parts)

TYPO_LIST = {
    "összételi"     : "összetételi",
    "összett"       : "összetett",
    "összetétett"   : "összetett",
    "erdetű"        : "eredetű",
    "eredeű"        : "eredetű",
    "kelekezésű"    : "keletkezésű"
}

def normalize_typos(text: str, typo_map: dict[str, str]) -> str:

    for wrong in sorted(typo_map, key=len, reverse=True):
        correct = typo_map[wrong]
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text, flags=re.IGNORECASE)
    return text

def remove_parentheses(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text)

FILLER_RE = re.compile(
    r"\b(különféle|azonban|ugyancsak|szintén|is|de|egy|a|az|keletkezett|létrehozott)\b",
    re.IGNORECASE
)

def remove_fillerwords(text: str) -> str:

    parts = re.split(r"(<i>.*?</i>|<b>.*?</b>)", text)

    for i, part in enumerate(parts):
        if not part.startswith("<i>") and not part.startswith("<b>"):
            parts[i] = FILLER_RE.sub("", part)

    return "".join(parts)

def normalize_probability_markers(text: str) -> str:

    return re.sub(
        r"\b(valószínűleg|valószínű|esetleg|talán|feltehetőleg|feltehetően|feltehető)\b",
        "<PROBABILITY>",
        text,
        flags=re.IGNORECASE,
    )

def replace_duplicate_summary(text: str) -> str:

    return re.sub(r"\bazonos(?:ak)?\b.*$", "<DUPLICATE>", text, flags=re.IGNORECASE)

def replace_bold_italic_with_lemma(text: str) -> str:

    if not text:
        return text

    # 1. lemma + bold homograph superscript
    text = re.sub(
        r"<b>\s*<i>.*?</i>\s*</b>\s*<b>[⁰¹²³⁴⁵⁶⁷⁸⁹0-9]+</b>",
        "<LEMMA><HOMOGRAPH>",
        text,
        flags=re.DOTALL,
    )

    # 2. plain bold+italic lemma
    text = re.sub(
        r"<b>\s*<i>.*?</i>\s*</b>",
        "<LEMMA>",
        text,
        flags=re.DOTALL,
    )

    # 3. plain italic lemma
    text = re.sub(
        r"<i>.*?</i>",
        "<LEMMA>",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_punctuation_and_spaces(text: str) -> str:
    """
    Normalize spacing while preserving commas.
    """
    # normalize spaces around commas
    text = re.sub(r"\s*,\s*", ", ", text)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

LANG_FORMS = [
    "{lang} nyelvekből",
    "{lang} nyelvből",
    "{lang}ból",
    "{lang}ből",
    "{lang} eredetű",
    "{lang}",
]

_LANG_REPLACEMENTS_CACHE: list[tuple[str, re.Pattern[str], str]] | None = None

def replace_lang_forms(text: str, lang: str, family: str | None = None) -> str:
    escaped = re.escape(lang)
    tag = f"<LANG:{family}>" if family else "<LANG>"
    for pattern in LANG_FORMS:
        pat = pattern.format(lang=escaped)
        text = re.sub(rf"\b{pat}\b" if " " not in pat else pat, tag, text)
    return text


def _build_language_replacements(lang_to_family: dict[str, str]) -> list[tuple[str, re.Pattern[str], str]]:
    special_lang_variants = (
        set(cardinal_slavic_variants_in_summaries)
        | set(csuvas_turkic_summaries)
        | set(donor_lang)
    )

    replacements: list[tuple[str, re.Pattern[str], str]] = []

    for variant in sorted(special_lang_variants, key=len, reverse=True):
        family = None
        if variant in set(cardinal_slavic_variants_in_summaries):
            family = "Slavic"
        elif variant in set(csuvas_turkic_summaries):
            family = "Turkic"

        escaped = re.escape(variant)
        tag = f"<LANG:{family}>" if family else "<LANG>"
        for pattern_template in LANG_FORMS:
            pat = pattern_template.format(lang=escaped)
            pattern = re.compile(rf"\b{pat}\b" if " " not in pat else pat)
            replacements.append((variant, pattern, tag))

    for lang in sorted(lang_to_family.keys(), key=len, reverse=True):
        if lang in special_lang_variants:
            continue

        escaped = re.escape(lang)
        family = lang_to_family.get(lang)
        tag = f"<LANG:{family}>" if family else "<LANG>"
        for pattern_template in LANG_FORMS:
            pat = pattern_template.format(lang=escaped)
            pattern = re.compile(rf"\b{pat}\b" if " " not in pat else pat)
            replacements.append((lang, pattern, tag))

    return replacements


def normalize_languages(text: str, lang_to_family: dict[str, str]) -> str:
    global _LANG_REPLACEMENTS_CACHE
    if _LANG_REPLACEMENTS_CACHE is None:
        _LANG_REPLACEMENTS_CACHE = _build_language_replacements(lang_to_family)

    for needle, pattern, tag in _LANG_REPLACEMENTS_CACHE:
        if needle not in text:
            continue
        text = pattern.sub(tag, text)

    return text

def normalize_punctuation_and_spaces(text: str) -> str:
    """
    Clean leftover punctuation/spaces after normalization.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_constituent_lexicalization_tags(text: str) -> str:
    return re.sub(
        r"<(CALQUE|COMPOUND|REDUPLICATION)>\s+"
        r"(előtagja|előtagjai|előtagjának|utótagja|utótagjai|utótagjának)"
        r"(?:\s+(önállósulásával|önállósodásával|önállósulása|önállósodása))?"
        r"(?=[\s,.;:!?]|$)",
        "<CONSTITUENT_LEXICALIZATION>",
        text,
        flags=re.IGNORECASE,
    )

def normalize_summary_text(text: str, lang_to_family: dict[str, str]) -> str:
    text = text.casefold()
    text = normalize_typos(text, TYPO_LIST)
    text = replace_duplicate_summary(text) # first replace 
    text = remove_fillerwords(text)
    text = replace_bold_italic_with_lemma(text)
    text = remove_parentheses(text)

    text = normalize_probability_markers(text)

    text = apply_normalization_patterns(text, NORMALIZATION_PATTERNS)
    text = normalize_constituent_lexicalization_tags(text)
    text = normalize_languages(text, lang_to_family)

    text = normalize_punctuation_and_spaces(text)

    return text

def mark_disputed_from_normalized_summary(etym: Dict[str, Any]) -> None:

    etym_sum_raw = (etym.get("etym_sum_raw") or "").strip()

    # Initialize flags to False by default
    etym["is_disputed_any"] = False
    etym["is_disputed_origin"] = False
    etym["is_disputed_formation"] = False
    etym["is_disputed_base_word"] = False
    etym["is_disputed_cognates"] = False
    etym["is_duplicate"] = False

    if not etym_sum_raw:
        return

    summary_raw = ""
    for block in etym.get("blocks", []):
        if block.get("symbol") == "SUMMARY":
            summary_raw = (block.get("text_raw") or "").strip()
            break

    if not summary_raw:
        summary_raw = (
            (etym.get("summary_raw") or "")
            or (etym.get("summary_plain") or "")
        ).strip()

    if not summary_raw:

        return

    normalized = normalize_summary_text(summary_raw, LANG_TO_FAMILY)
    etym["summary_normalized"] = normalized
    etym["is_duplicate"] = "<DUPLICATE>" in normalized

    if "<DISPUTED_" not in normalized:
        return

    text = etym_sum_raw

    idx = text.lower().find(summary_raw.lower())

    if idx != -1:

        after_summary = text[idx + len(summary_raw):]
    else:

        after_summary = text

    after_summary = re.sub(
        r'(?:\|\s*)?<b>\s*(\d+)\.\s*</b>\s*',
        r'| \1. ',
        after_summary,
    )
    has_numbered_analysis = bool(re.search(r"\|\s*1\.", after_summary))

    if not has_numbered_analysis:

        return

    etym["is_disputed_any"] = True

    # Fine-grained flags from the tags:
    if "<DISPUTED_ORIGIN>" in normalized:
        etym["is_disputed_origin"] = True
    if "<DISPUTED_FORMATION>" in normalized:
        etym["is_disputed_formation"] = True
    if "<DISPUTED_BASE>" in normalized:
        etym["is_disputed_base_word"] = True
    if "<DISPUTED_COGNATES>" in normalized:
        etym["is_disputed_cognates"] = True
