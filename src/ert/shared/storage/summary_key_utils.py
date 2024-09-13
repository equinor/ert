from enum import Enum, auto
from typing import List

special_keys = [
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


class SummaryKeyType(Enum):
    INVALID = auto()
    FIELD = auto()
    REGION = auto()
    GROUP = auto()
    WELL = auto()
    SEGMENT = auto()
    BLOCK = auto()
    AQUIFER = auto()
    COMPLETION = auto()
    NETWORK = auto()
    REGION_2_REGION = auto()
    LOCAL_BLOCK = auto()
    LOCAL_COMPLETION = auto()
    LOCAL_WELL = auto()
    MISC = auto()

    @staticmethod
    def determine_key_type(key: str) -> "SummaryKeyType":
        if key in special_keys:
            return SummaryKeyType.MISC

        if key.startswith("L"):
            secondary = key[1] if len(key) > 1 else ""
            return {
                "B": SummaryKeyType.LOCAL_BLOCK,
                "C": SummaryKeyType.LOCAL_COMPLETION,
                "W": SummaryKeyType.LOCAL_WELL,
            }.get(secondary, SummaryKeyType.MISC)

        if key.startswith("R"):
            if len(key) == 3 and key[2] == "F":
                return SummaryKeyType.REGION_2_REGION
            if key == "RNLF":
                return SummaryKeyType.REGION_2_REGION
            if key == "RORFR":
                return SummaryKeyType.REGION
            if len(key) >= 4 and key[2] == "F" and key[3] in {"T", "R"}:
                return SummaryKeyType.REGION_2_REGION
            if len(key) >= 5 and key[3] == "F" and key[4] in {"T", "R"}:
                return SummaryKeyType.REGION_2_REGION
            return SummaryKeyType.REGION

        # default cases or miscellaneous if not matched
        return {
            "A": SummaryKeyType.AQUIFER,
            "B": SummaryKeyType.BLOCK,
            "C": SummaryKeyType.COMPLETION,
            "F": SummaryKeyType.FIELD,
            "G": SummaryKeyType.GROUP,
            "N": SummaryKeyType.NETWORK,
            "S": SummaryKeyType.SEGMENT,
            "W": SummaryKeyType.WELL,
        }.get(key[0], SummaryKeyType.MISC)


def _match_keyword_vector(start: int, rate_keys: List[str], keyword: str) -> bool:
    if len(keyword) < start:
        return False
    return any(keyword[start:].startswith(key) for key in rate_keys)


def _match_keyword_string(start: int, rate_string: str, keyword: str) -> bool:
    if len(keyword) < start:
        return False
    return keyword[start:].startswith(rate_string)


def is_rate(key: str) -> bool:
    key_type = SummaryKeyType.determine_key_type(key)
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

    if key_type == SummaryKeyType.REGION_2_REGION:
        # Region to region rates are identified by R*FR or R**FR
        if _match_keyword_string(2, "FR", key):
            return True
        return _match_keyword_string(3, "FR", key)

    return False
