import re
from . import regexes as rx
from . import normalization as nm
from . import ueszweb_abbreviations as ab

ETY_BLOCK_SYMBOLS = list(ab.ETY_BLOCK_ROLE_MAP.keys())

import re

def extract_etymological_summary_from_chunk(etym_sum_raw: str, etym_sum_plain: str) -> dict:

    def _cut_intro(text: str) -> str:
        if not text:
            return ""

        t = text.strip()

        t = re.sub(r'^\s*■\s*', '', t, count=1)

        if "|" in t:
            t = t.split("|", 1)[0].strip()

        t = " ".join(t.split()).strip()
        return t

    raw_summary = _cut_intro(etym_sum_raw)
    plain_summary = _cut_intro(etym_sum_plain).rstrip(" .;:")

    if not plain_summary:
        return {"summary_raw": "", "summary_plain": ""}

    if plain_summary.startswith("□"):
        return {"summary_raw": "", "summary_plain": ""}

    if re.match(r'^a szócsaládhoz még\b', plain_summary, flags=re.IGNORECASE):
        return {"summary_raw": "", "summary_plain": ""}

    return {
        "summary_raw": raw_summary,
        "summary_plain": plain_summary,
    }

