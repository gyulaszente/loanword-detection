import pandas as pd

SUFFIX_VARIANTS = {
    "dik":   "dik/adik/odik/edik/ödik/ódik/ődik",
    "adik":  "dik/adik/odik/edik/ödik/ódik/ődik",
    "edik":  "dik/adik/odik/edik/ödik/ódik/ődik",
    "ödik":  "dik/adik/odik/edik/ödik/ódik/ődik",
    "ódik":  "dik/adik/odik/edik/ödik/ódik/ődik",
    "odik":  "dik/adik/odik/edik/ödik/ódik/ődik",
    "ődik":  "dik/adik/odik/edik/ödik/ódik/ődik",

    "ik":    "ik/lik/szik",
    "lik":   "ik/lik/szik",
    "szik":  "ik/lik/szik",


    "zik":   "zik/azik/ozik/ezik/özik",
    "azik":  "zik/azik/ozik/ezik/özik",
    "ozik":  "zik/azik/ozik/ezik/özik",
    "ezik":  "zik/azik/ozik/ezik/özik",
    "özik":  "zik/azik/ozik/ezik/özik",


    "kozik": "kozik/kezik/közik",
    "kezik": "kozik/kezik/közik",
    "közik": "kozik/kezik/közik",


    "kodik": "kodik/kedik/ködik/skodik/skedik/sködik",
    "kedik": "kodik/kedik/ködik/skodik/skedik/sködik",
    "ködik": "kodik/kedik/ködik/skodik/skedik/sködik",
    "skodik": "kodik/kedik/ködik/skodik/skedik/sködik",
    "skedik": "kodik/kedik/ködik/skodik/skedik/sködik",
    "sködik": "kodik/kedik/ködik/skodik/skedik/sködik",
    

    "ú":  "ú/ű",
    "ű":  "ú/ű",

    "akodik": "akodik/akozik",
    "akozik": "akodik/akozik",

    "mány":   "mány/mény/omány/emény",
    "mény":   "mány/mény/omány/emény",
    "omány":  "mány/mény/omány/emény",
    "emény":  "mány/mény/omány/emény",
    "vány":   "vány/vény",
    "vény":   "vány/vény",

    "ság":   "ság/ség",
    "ség":   "ság/ség",

    "ságos": "ságos/séges",
    "séges": "ságos/séges",

    "al":    "al/el/ol/öl/ul/ül",
    "el":    "al/el/ol/öl/ul/ül",
    "ol":    "al/el/ol/öl/ul/ül",
    "öl":    "al/el/ol/öl/ul/ül",
    "ul":    "al/el/ol/öl/ul/ül",
    "ül":    "al/el/ol/öl/ul/ül",


    "ászkodik": "ászkodik/észkedik",
    "észkedik": "ászkodik/észkedik",
    "ászkozik": "ászkodik/észkedik",
    "ékodik":   "ékodik/ékszik",
    "ékszik":   "ékodik/ékszik",

    "atos":  "atos/etes/ős",
    "etes":  "atos/etes/ős",
    "ős":    "atos/etes/ős",

    "s":     "s/os/es/as/ös",
    "os":    "s/os/es/as/ös",
    "es":    "s/os/es/as/ös",
    "as":    "s/os/es/as/ös",
    "ös":    "s/os/es/as/ös",

    "ás":    "ás/és",
    "és":    "ás/és",

    "at":    "at/et",
    "et":    "at/et",

    "z":     "z/az/oz/ez/öz",
    "az":    "z/az/oz/ez/öz",
    "oz":    "z/az/oz/ez/öz",
    "ez":    "z/az/oz/ez/öz",
    "öz":    "z/az/oz/ez/öz",

    "alom":   "alom/elem/odalom/edelem",
    "elem":   "alom/elem/odalom/edelem",
    "odalom": "alom/elem/odalom/edelem",
    "edelem": "alom/elem/odalom/edelem",

    "tat":   "tat/tet",
    "tet":   "tat/tet",

    "zat":   "zat/ozat/zet/ezet",
    "ozat":  "zat/ozat/zet/ezet",
    "zet":   "zat/ozat/zet/ezet",
    "ezet":  "zat/ozat/zet/ezet",

    "lat":   "lat/let",
    "let":   "lat/let",

    "lás":   "lás/lés/olás/elés",
    "lés":   "lás/lés/olás/elés",
    "olás":  "lás/lés/olás/elés",
    "elés":  "lás/lés/olás/elés",

    "nok":   "nok/nök",
    "nök":   "nok/nök",

    "talan": "telen/talan/atlan/etlen/tlan/tlen",
    "telen": "telen/talan/atlan/etlen/tlan/tlen",
    "tlan":  "telen/talan/atlan/etlen/tlan/tlen",
    "tlen":  "telen/talan/atlan/etlen/tlan/tlen",
    "atlan": "telen/talan/atlan/etlen/tlan/tlen",
    "etlen": "telen/talan/atlan/etlen/tlan/tlen",
    "len":   "telen/talan/atlan/etlen/tlan/tlen",
    "lan":   "telen/talan/atlan/etlen/tlan/tlen",

    "ász":   "ász/ész",
    "ész":   "ász/ész",

    "ka":    "ka/ke",
    "ke":    "ka/ke",
    "cska":  "cska/cske/acska/ecske",
    "cske":  "cska/cske/acska/ecske",
    "acska": "cska/cske/acska/ecske",
    "ecske": "cska/cske/acska/ecske",

    "ó":     "ó/ő",
    "ő":     "ó/ő",

    "ít":    "ít",
    "i":     "i",

    "ék":    "ék/dék/adék/edék",
    "dék":   "ék/dék/adék/edék",
    "adék":  "ék/dék/adék/edék",
    "edék":  "ék/dék/adék/edék",

    "hat":   "hat/het",
    "het":   "hat/het",
    "ható":  "ható/hető",
    "hető":  "ható/hető",

    "gat":   "gat/get/ogat/eget/öget",
    "get":   "gat/get/ogat/eget/öget",
    "ogat":  "gat/get/ogat/eget/öget",
    "eget":  "gat/get/ogat/eget/öget",
    "öget":  "gat/get/ogat/eget/öget",

    "t":     "t/ott/ett/ött",
    "ott":   "t/ott/ett/ött",
    "ett":   "t/ott/ett/ött",
    "ött":   "t/ott/ett/ött",

    "d":     "d/od/ed/öd",
    "od":    "d/od/ed/öd",
    "ed":    "d/od/ed/öd",
    "öd":    "d/od/ed/öd",
}


def normalize_suffix(s):
    if pd.isna(s) or not s:
        return s

    s = str(s).strip()

    if s in SUFFIX_VARIANTS:
        return SUFFIX_VARIANTS[s]

    if "/" in s:
        for part in s.split("/"):
            part = part.strip()
            if part in SUFFIX_VARIANTS:
                return SUFFIX_VARIANTS[part]

    return s


""" based partially on the emmorph code list """
SUFFIX_GLOSSES = {
    "s/os/es/as/ös":                                "adjectival suffix: 'having X; characterized by X'",
    "ság/ség":                                      "abstract noun suffix: '-ness, -ity'",
    "z/az/oz/ez/öz":                                "verb-forming suffix: 'to do/use X'",
    "ás/és":                                     "action/result noun suffix: '-ing, act of X'",
    "ó/ő":                                          "participle/agentive suffix: '-ing; one that does X'",
    "telen/talan/atlan/etlen/tlan/tlen":            "privative suffix: 'without X, -less'",
    "i":                                            "relational adjective suffix: 'of/from X'",
    "ít":                                           "causative/factitive verb suffix: 'make/become X'",
    "al/el/ol/öl/ul/ül":                        "verb-forming suffix: 'to act/use/become X'",
    "at/et":                                        "nominal/result suffix: 'result/product of X'",
    "ik/lik/szik":                                      "ik-verb marker: reflexive/intransitive citation-form suffix",
    "zik/azik/ozik/ezik/özik":                  "verb-forming suffix: 'to engage in/perform X' (intransitive)",
    "dik/adik/odik/edik/ödik/ódik/ődik":            "verb-forming suffix: inchoative/adjectival verbalizer; ordinal",
    "kodik/kedik/ködik/skodik/skedik/sködik":       "verb-forming suffix: 'behave as X; engage in X'",
    "zat/ozat/zet/ezet":                            "result/collective noun suffix",
    "lat/let":                                      "nominal suffix: 'result/state of X'",
    "ék/dék/adék/edék":                         "nominal suffix: 'result/material/remnant'",
    "kozik/kezik/közik":                        "reflexive verb suffix group",
    "mány/mény/omány/emény":                    "nominal suffix: 'result/product of X'",
}
