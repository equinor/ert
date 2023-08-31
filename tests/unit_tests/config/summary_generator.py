"""
Implements a hypothesis strategy for unified summary files
(.SMSPEC and .UNSMRY) without any optional fields.
See https://opm-project.org/?page_id=955
"""
from dataclasses import astuple, dataclass
from datetime import datetime, timedelta
from enum import Enum, unique
from typing import Any, List, Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import resfo
from hypothesis import assume
from hypothesis.extra.numpy import from_dtype
from pydantic import PositiveInt, conint
from typing_extensions import Self

from ert.config._read_summary import SPECIAL_KEYWORDS

from .egrid_generator import GrdeclKeyword

"""
See section 11.2 in opm flow reference manual 2022-10
for definition of summary variable names.
"""

inter_region_summary_variables = [
    "RGFR",
    "RGFR+",
    "RGFR-",
    "RGFT",
    "RGFT+",
    "RGFT-",
    "RGFTG",
    "RGFTL",
    "ROFR",
    "ROFR+",
    "ROFR-",
    "ROFT",
    "ROFT+",
    "ROFT-",
    "ROFTG",
    "ROFTL",
    "RWFR",
    "RWFR+",
    "RWFR-",
    "RWFT",
    "RWFT+",
    "RWFT-",
    "RCFT",
    "RSFT",
    "RNFT",
]


@st.composite
def root_memnonic(draw):
    first_character = draw(st.sampled_from("ABFGRWCS"))
    if first_character == "A":
        second_character = draw(st.sampled_from("ALN"))
        third_character = draw(st.sampled_from("QL"))
        fourth_character = draw(st.sampled_from("RT"))
        return first_character + second_character + third_character + fourth_character
    else:
        second_character = draw(st.sampled_from("OWGVLPT"))
        third_character = draw(st.sampled_from("PIF"))
        fourth_character = draw(st.sampled_from("RT"))
        # local = draw(st.sampled_from(["", "L"])) if first_character in "BCW" else ""
        return first_character + second_character + third_character + fourth_character


@st.composite
def summary_variables(draw):
    kind = draw(
        st.sampled_from(
            [
                "special",
                "network",
                "exceptions",
                "directional",
                "up_or_down",
                "mnemonic",
                "segment",
                "well",
                "region2region",
                "memnonic",
                "region",
            ]
        )
    )
    if kind == "special":
        return draw(st.sampled_from(SPECIAL_KEYWORDS))
    if kind == "exceptions":
        return draw(
            st.sampled_from(
                ["BAPI", "BOSAT", "BPR", "FAQR", "FPR", "FWCT", "WBHP", "WWCT", "ROFR"]
            )
        )
    elif kind == "directional":
        direction = draw(st.sampled_from("IJK"))
        return (
            draw(st.sampled_from(["FLOO", "VELG", "VELO", "FLOW", "VELW"])) + direction
        )
    elif kind == "up_or_down":
        dimension = draw(st.sampled_from("XYZRT"))
        direction = draw(st.sampled_from(["", "-"]))
        return draw(st.sampled_from(["GKR", "OKR", "WKR"])) + dimension + direction
    elif kind == "network":
        root = draw(root_memnonic())
        return "N" + root
    elif kind == "segment":
        return draw(
            st.sampled_from(["SALQ", "SFR", "SGFR", "SGFRF", "SGFRS", "SGFTA", "SGFT"])
        )
    elif kind == "well":
        return draw(
            st.one_of(
                st.builds(lambda r: "W" + r, root_memnonic()),
                st.sampled_from(
                    [
                        "WBHP",
                        "WBP5",
                        "WBP4",
                        "WBP9",
                        "WBP",
                        "WBHPH",
                        "WBHPT",
                        "WPIG",
                        "WPIL",
                        "WPIO",
                        "WPI5",
                    ]
                ),
            )
        )
    elif kind == "region2region":
        return draw(st.sampled_from(inter_region_summary_variables))
    elif kind == "region":
        return draw(st.builds(lambda r: "R" + r, root_memnonic()))
    else:
        return draw(root_memnonic())


unit_names = st.sampled_from(
    ["SM3/DAY", "BARSA", "SM3/SM3", "FRACTION", "DAYS", "HOURS", "SM3"]
)

names = st.text(
    min_size=8, max_size=8, alphabet=st.characters(min_codepoint=65, max_codepoint=90)
)


@unique
class UnitSystem(Enum):
    METRIC = 1
    FIELD = 2
    LAB = 3

    def to_ecl(self):
        return self.value


@unique
class Simulator(Enum):
    ECLIPSE_100 = 100
    ECLIPSE_300 = 300
    ECLIPSE_300_THERMAL = 500
    INTERSECT = 700
    FRONTSIM = 800

    def to_ecl(self):
        return self.value


@dataclass
class SmspecIntehead(GrdeclKeyword):
    unit: UnitSystem
    simulator: Simulator


@dataclass
class Date:
    day: conint(ge=1, le=31)  # type: ignore
    month: conint(ge=1, le=12)  # type: ignore
    year: conint(gt=1901, lt=2038)  # type: ignore
    hour: conint(ge=0, lt=24)  # type: ignore
    minutes: conint(ge=0, lt=60)  # type: ignore
    micro_seconds: conint(ge=0, lt=60000000)  # type: ignore

    def to_ecl(self):
        return astuple(self)

    def to_datetime(self) -> datetime:
        return datetime(
            year=self.year,
            month=self.month,
            day=self.day,
            hour=self.hour,
            minute=self.minutes,
            second=self.micro_seconds // 10**6,
            microsecond=self.micro_seconds % 10**6,
        )

    @classmethod
    def from_datetime(cls, dt: datetime) -> Self:
        return cls(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hour=dt.hour,
            minutes=dt.minute,
            micro_seconds=dt.second * 10**6 + dt.microsecond,
        )


@dataclass
class Smspec:
    intehead: SmspecIntehead
    restart: str
    num_keywords: PositiveInt
    nx: PositiveInt
    ny: PositiveInt
    nz: PositiveInt
    restarted_from_step: PositiveInt
    keywords: List[str]
    well_names: List[str]
    region_numbers: List[int]
    units: List[str]
    start_date: Date
    lgr_names: Optional[List[str]] = None
    lgrs: Optional[List[str]] = None
    numlx: Optional[List[PositiveInt]] = None
    numly: Optional[List[PositiveInt]] = None
    numlz: Optional[List[PositiveInt]] = None

    def to_ecl(self) -> List[Tuple[str, Any]]:
        # The restart field contains 9 strings of length 8 which
        # should contain the name of the file restarted from.
        # If shorter than 72 characters (most likely), the rest
        # are spaces. (opm manual table F.44, keyword name RESTART)
        restart = self.restart.ljust(72, " ")
        restart_list = [restart[i * 8 : i * 8 + 8] for i in range(9)]
        return (
            [
                ("INTEHEAD", np.array(self.intehead.to_ecl(), dtype=np.int32)),
                ("RESTART ", restart_list),
                (
                    "DIMENS  ",
                    np.array(
                        [
                            self.num_keywords,
                            self.nx,
                            self.ny,
                            self.nz,
                            0,
                            self.restarted_from_step,
                        ],
                        dtype=np.int32,
                    ),
                ),
                ("KEYWORDS", [kw.ljust(8) for kw in self.keywords]),
                ("WGNAMES ", self.well_names),
                ("NUMS    ", np.array(self.region_numbers, dtype=np.int32)),
                ("UNITS   ", self.units),
                ("STARTDAT", np.array(self.start_date.to_ecl(), dtype=np.int32)),
            ]
            + ([("LGRS    ", self.lgrs)] if self.lgrs is not None else [])
            + ([("LGRNAMES", self.lgr_names)] if self.lgr_names is not None else [])
            + ([("NUMLX   ", self.numlx)] if self.numlx is not None else [])
            + ([("NUMLY   ", self.numly)] if self.numly is not None else [])
            + ([("NUMLZ   ", self.numlz)] if self.numlz is not None else [])
        )

    def to_file(self, filelike, file_format: resfo.Format = resfo.Format.UNFORMATTED):
        resfo.write(filelike, self.to_ecl(), file_format)


positives = from_dtype(np.dtype(np.int32), min_value=1, max_value=10000)
small_ints = from_dtype(np.dtype(np.int32), min_value=1, max_value=10)


@st.composite
def smspecs(
    draw,
    sum_keys,
    start_date,
):
    """
    Strategy for smspec that ensures that the TIME parameter, as required by
    ert, is in the parameters list.
    """
    use_locals = draw(st.booleans())
    sum_keys = draw(sum_keys)
    n = len(sum_keys) + 1
    nx = draw(small_ints)
    ny = draw(small_ints)
    nz = draw(small_ints)
    keywords = ["TIME    "] + sum_keys
    units = ["DAYS    "] + draw(st.lists(unit_names, min_size=n - 1, max_size=n - 1))
    well_names = [":+:+:+:+"] + draw(st.lists(names, min_size=n - 1, max_size=n - 1))
    if use_locals:  # use local
        lgrs = draw(st.lists(names, min_size=n, max_size=n))
        numlx = draw(st.lists(small_ints, min_size=n, max_size=n))
        numly = draw(st.lists(small_ints, min_size=n, max_size=n))
        numlz = draw(st.lists(small_ints, min_size=n, max_size=n))
        lgr_names = list(set(lgrs))
    else:
        lgrs = None
        numlx = None
        numly = None
        numlz = None
        lgr_names = None
    region_numbers = [-32676] + draw(
        st.lists(
            from_dtype(np.dtype(np.int32), min_value=1, max_value=nx * ny * nz),
            min_size=len(sum_keys),
            max_size=len(sum_keys),
        )
    )
    return draw(
        st.builds(
            Smspec,
            nx=st.just(nx),
            ny=st.just(ny),
            nz=st.just(nz),
            # restarted_from_step is hardcoded to 0 because
            # of a bug in enkf_obs where it assumes that
            # ecl_sum_get_first_report_step is always 1
            restarted_from_step=st.just(0),
            num_keywords=st.just(n),
            restart=names,
            keywords=st.just(keywords),
            well_names=st.just(well_names),
            lgrs=st.just(lgrs),
            numlx=st.just(numlx),
            numly=st.just(numly),
            numlz=st.just(numlz),
            lgr_names=st.just(lgr_names),
            region_numbers=st.just(region_numbers),
            units=st.just(units),
            start_date=start_date,
        )
    )


@dataclass
class SummaryMiniStep:
    mini_step: int
    params: List[float]

    def to_ecl(self):
        return [
            ("MINISTEP", np.array([self.mini_step], dtype=np.int32)),
            ("PARAMS  ", np.array(self.params, dtype=np.float32)),
        ]


@dataclass
class SummaryStep:
    seqnum: int
    ministeps: List[SummaryMiniStep]

    def to_ecl(self):
        return [("SEQHDR  ", np.array([self.seqnum], dtype=np.int32))] + [
            i for ms in self.ministeps for i in ms.to_ecl()
        ]


@dataclass
class Unsmry:
    steps: List[SummaryStep]

    def to_ecl(self):
        a = [i for step in self.steps for i in step.to_ecl()]
        return a

    def to_file(self, filelike, file_format: resfo.Format = resfo.Format.UNFORMATTED):
        resfo.write(filelike, self.to_ecl(), file_format)


positive_floats = from_dtype(
    np.dtype(np.float32),
    min_value=np.float32(0.1),
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def unsmrys(
    draw,
    num_params,
    report_steps,
    mini_steps,
    days,
):
    """
    Simplified strategy for unsmrys that will generate a summary file
    with one ministep per report step.
    """
    n = num_params
    rs = sorted(draw(report_steps))
    ms = sorted(draw(mini_steps), reverse=True)
    ds = sorted(draw(days), reverse=True)
    steps = []
    for r in rs:
        minis = [
            SummaryMiniStep(
                ms.pop(),
                [ds.pop()] + draw(st.lists(positive_floats, min_size=n, max_size=n)),
            )
        ]
        steps.append(SummaryStep(r, minis))
    return Unsmry(steps)


@st.composite
def summaries(draw):
    sum_keys = draw(st.lists(summary_variables(), min_size=1))
    first_date = draw(
        st.datetimes(
            min_value=datetime.strptime("1999-1-1", "%Y-%m-%d"),
            max_value=datetime.strptime("3000-1-1", "%Y-%m-%d"),
        )
    )
    smspec = draw(
        smspecs(
            sum_keys=st.just(sum_keys),
            start_date=st.just(
                Date(
                    year=first_date.year,
                    month=first_date.month,
                    day=first_date.day,
                    hour=first_date.hour,
                    minutes=first_date.minute,
                    micro_seconds=first_date.second * 10**6 + first_date.microsecond,
                )
            ),
        )
    )
    assume(
        len(set(zip(smspec.keywords, smspec.region_numbers, smspec.well_names)))
        == len(smspec.keywords)
    )
    dates = [0.0] + draw(
        st.lists(
            st.floats(
                min_value=0.1,
                max_value=250_000,  # in days ~= 685 years
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=100,
        )
    )
    try:
        _ = first_date + timedelta(days=max(dates))
    except (ValueError, OverflowError):  # datetime has a max year
        assume(False)

    ds = sorted(dates, reverse=True)
    steps = []
    i = 0
    j = 0
    while len(ds) > 0:
        minis = []
        for _ in range(draw(st.integers(min_value=1, max_value=len(ds)))):
            minis.append(
                SummaryMiniStep(
                    i,
                    [ds.pop()]
                    + draw(
                        st.lists(
                            positive_floats,
                            min_size=len(sum_keys),
                            max_size=len(sum_keys),
                        )
                    ),
                )
            )
            i += 1
        steps.append(SummaryStep(j, minis))
        j += 1
    return smspec, Unsmry(steps)
