from __future__ import annotations

import re
from enum import Enum, auto
from typing import (
    List,
)

SPECIAL_KEYWORDS = [
    "NAIMFRAC",
    "NBAKFL",
    "NBYTOT",
    "NCPRLINS",
    "NEWTFL",
    "NEWTON",
    "NLINEARP",
    "NLINEARS",
    "NLINSMAX",
    "NLINSMIN",
    "NLRESMAX",
    "NLRESSUM",
    "NMESSAGE",
    "NNUMFL",
    "NNUMST",
    "NTS",
    "NTSECL",
    "NTSMCL",
    "NTSPCL",
    "ELAPSED",
    "MAXDPR",
    "MAXDSO",
    "MAXDSG",
    "MAXDSW",
    "STEPTYPE",
    "WNEWTON",
]


class SummaryKeyType(Enum):
    AQUIFER = auto()
    BLOCK = auto()
    COMPLETION = auto()
    FIELD = auto()
    GROUP = auto()
    LOCAL_BLOCK = auto()
    LOCAL_COMPLETION = auto()
    LOCAL_WELL = auto()
    NETWORK = auto()
    SEGMENT = auto()
    WELL = auto()
    REGION = auto()
    INTER_REGION = auto()
    OTHER = auto()

    @classmethod
    def from_keyword(cls, summary_keyword: str) -> SummaryKeyType:
        KEYWORD_TYPE_MAPPING = {
            "A": cls.AQUIFER,
            "B": cls.BLOCK,
            "C": cls.COMPLETION,
            "F": cls.FIELD,
            "G": cls.GROUP,
            "LB": cls.LOCAL_BLOCK,
            "LC": cls.LOCAL_COMPLETION,
            "LW": cls.LOCAL_WELL,
            "N": cls.NETWORK,
            "S": cls.SEGMENT,
            "W": cls.WELL,
        }
        if not summary_keyword:
            raise ValueError("Got empty summary keyword")
        if any(special in summary_keyword for special in SPECIAL_KEYWORDS):
            return cls.OTHER
        if summary_keyword[0] in KEYWORD_TYPE_MAPPING:
            return KEYWORD_TYPE_MAPPING[summary_keyword[0]]
        if summary_keyword[0:2] in KEYWORD_TYPE_MAPPING:
            return KEYWORD_TYPE_MAPPING[summary_keyword[0:2]]
        if summary_keyword == "RORFR":
            return cls.REGION

        if any(
            re.fullmatch(pattern, summary_keyword)
            for pattern in [r"R.FT.*", r"R..FT.*", r"R.FR.*", r"R..FR.*", r"R.F"]
        ):
            return cls.INTER_REGION
        if summary_keyword[0] == "R":
            return cls.REGION

        return cls.OTHER


rate_keys = [
    "OPR",
    "OIR",
    "OVPR",
    "OVIR",
    "OFR",
    "OPP",
    "OPI",
    "OMR",
    "GPR",
    "GIR",
    "GVPR",
    "GVIR",
    "GFR",
    "GPP",
    "GPI",
    "GMR",
    "WGPR",
    "WGIR",
    "WPR",
    "WIR",
    "WVPR",
    "WVIR",
    "WFR",
    "WPP",
    "WPI",
    "WMR",
    "LPR",
    "LFR",
    "VPR",
    "VIR",
    "VFR",
    "GLIR",
    "RGR",
    "EGR",
    "EXGR",
    "SGR",
    "GSR",
    "FGR",
    "GIMR",
    "GCR",
    "NPR",
    "NIR",
    "CPR",
    "CIR",
    "SIR",
    "SPR",
    "TIR",
    "TPR",
    "GOR",
    "WCT",
    "OGR",
    "WGR",
    "GLR",
]

seg_rate_keys = [
    "OFR",
    "GFR",
    "WFR",
    "CFR",
    "SFR",
    "TFR",
    "CVPR",
    "WCT",
    "GOR",
    "OGR",
    "WGR",
]


def _match_keyword_vector(start: int, rate_keys: List[str], keyword: str) -> bool:
    if len(keyword) < start:
        return False
    return any(keyword[start:].startswith(key) for key in rate_keys)


def _match_keyword_string(start: int, rate_string: str, keyword: str) -> bool:
    if len(keyword) < start:
        return False
    return keyword[start:].startswith(rate_string)


def is_rate(key: str) -> bool:
    key_type = SummaryKeyType.from_keyword(key)
    if key_type in {
        SummaryKeyType.WELL,
        SummaryKeyType.GROUP,
        SummaryKeyType.FIELD,
        SummaryKeyType.REGION,
        SummaryKeyType.COMPLETION,
        SummaryKeyType.LOCAL_WELL,
        SummaryKeyType.LOCAL_COMPLETION,
        SummaryKeyType.NETWORK,
    }:
        if key_type in {
            SummaryKeyType.LOCAL_WELL,
            SummaryKeyType.LOCAL_COMPLETION,
            SummaryKeyType.NETWORK,
        }:
            return _match_keyword_vector(2, rate_keys, key)
        return _match_keyword_vector(1, rate_keys, key)

    if key_type == SummaryKeyType.SEGMENT:
        return _match_keyword_vector(1, seg_rate_keys, key)

    if key_type == SummaryKeyType.INTER_REGION:
        # Region to region rates are identified by R*FR or R**FR
        if _match_keyword_string(2, "FR", key):
            return True
        return _match_keyword_string(3, "FR", key)

    return False
