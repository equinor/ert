from __future__ import annotations

import re
from enum import Enum, auto

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
    """Summary keys are divided into types based on summary variable name.

    see :ref:`SUMMARY  <summary>` for in keywords.rst for details on summary
    variables and keys.

    """

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
    def from_variable(cls, summary_variable: str) -> SummaryKeyType:
        """Returns the type corresponding to the given summary variable

        >>> SummaryKeyType.from_variable("FOPR").name
        'FIELD'
        >>> SummaryKeyType.from_variable("LWWIT").name
        'LOCAL_WELL'
        """
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
        if not summary_variable:
            raise ValueError("Got empty summary keyword")
        if any(special in summary_variable for special in SPECIAL_KEYWORDS):
            return cls.OTHER
        if summary_variable[0] in KEYWORD_TYPE_MAPPING:
            return KEYWORD_TYPE_MAPPING[summary_variable[0]]
        if summary_variable[0:2] in KEYWORD_TYPE_MAPPING:
            return KEYWORD_TYPE_MAPPING[summary_variable[0:2]]
        if summary_variable == "RORFR":
            return cls.REGION

        if any(
            re.fullmatch(pattern, summary_variable)
            for pattern in [r"R.FT.*", r"R..FT.*", r"R.FR.*", r"R..FR.*", r"R.F"]
        ):
            return cls.INTER_REGION
        if summary_variable[0] == "R":
            return cls.REGION

        return cls.OTHER


def is_rate(summary_variable: str) -> bool:
    """Whether the given summary variable is a rate.

    See `opm flow reference manual
    <https://opm-project.org/wp-content/uploads/2023/06/OPM_Flow_Reference_Manual_2023-04_Rev-0_Reduced.pdf>`
    table 11.4 for details.
    """
    match SummaryKeyType.from_variable(summary_variable):
        case (
            SummaryKeyType.WELL
            | SummaryKeyType.GROUP
            | SummaryKeyType.FIELD
            | SummaryKeyType.REGION
            | SummaryKeyType.COMPLETION
        ):
            return _match_rate_root(1, _rate_roots, summary_variable)
        case (
            SummaryKeyType.LOCAL_WELL
            | SummaryKeyType.LOCAL_COMPLETION
            | SummaryKeyType.NETWORK
        ):
            return _match_rate_root(2, _rate_roots, summary_variable)
        case SummaryKeyType.SEGMENT:
            return _match_rate_root(1, _segment_rate_roots, summary_variable)
        case SummaryKeyType.INTER_REGION:
            # Region to region rates are identified by R*FR or R**FR
            return _match_rate_root(2, ["FR"], summary_variable) or _match_rate_root(
                3, ["FR"], summary_variable
            )

    return False


__all__ = ["SummaryKeyType", "is_rate"]


_rate_roots = [  # see opm-flow-manual 2023-04 table 11.8, 11.9 & 11.14
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
    "GOR",  # dimensionless but considered a rate, as the ratio of two rates
    "WCT",  # dimensionless but considered a rate, as the ratio of two rates
    "OGR",  # dimensionless but considered a rate, as the ratio of two rates
    "WGR",  # dimensionless but considered a rate, as the ratio of two rates
    "GLR",  # dimensionless but considered a rate, as the ratio of two rates
]

_segment_rate_roots = [  # see opm-flow-manual 2023-04 table 11.19
    "OFR",
    "GFR",
    "WFR",
    "CFR",
    "SFR",
    "TFR",
    "CVPR",
    "WCT",  # dimensionless but considered a rate, as the ratio of two rates
    "GOR",  # dimensionless but considered a rate, as the ratio of two rates
    "OGR",  # dimensionless but considered a rate, as the ratio of two rates
    "WGR",  # dimensionless but considered a rate, as the ratio of two rates
]


def _match_rate_root(start: int, rate_roots: list[str], keyword: str) -> bool:
    if len(keyword) < start:
        return False
    return any(keyword[start:].startswith(key) for key in rate_roots)
