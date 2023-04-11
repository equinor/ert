"""
Implements a hypothesis strategy for unified summary files
(.SMSPEC and .UNSMRY) without any optional fields.
See https://opm-project.org/?page_id=955
"""
from dataclasses import astuple, dataclass
from enum import Enum, unique
from typing import Any, List, Tuple

import ecl_data_io
import hypothesis.strategies as st
import numpy as np
from pydantic import PositiveInt, conint

from .egrid_generator import GrdeclKeyword


@st.composite
def summary_variables(draw):
    """
    Generator for summary variable mnemonic, See
    section 11.2.1 in opm flow reference manual 2022-10.
    """
    first_character = draw(st.sampled_from("ABFGRWCS"))
    if first_character == "A":
        second_character = draw(st.sampled_from("ALN"))
        third_character = draw(st.sampled_from("QL"))
        fourth_character = draw(st.sampled_from("RT"))
        return first_character + second_character + third_character + fourth_character

    kind = draw(st.sampled_from([1, 2, 3, 4]))
    if kind == 1:
        return draw(
            st.sampled_from(
                ["BAPI", "BOSAT", "BPR", "FAQR", "FPR", "FWCT", "WBHP", "WWCT"]
            )
        )
    elif kind == 2:
        direction = draw(st.sampled_from("IJK"))
        return (
            draw(st.sampled_from(["FLOO", "VELG", "VELO", "FLOW", "VELW"])) + direction
        )
    elif kind == 3:
        dimension = draw(st.sampled_from("XYZRT"))
        direction = draw(st.sampled_from(["", "-"]))
        return draw(st.sampled_from(["GKR", "OKR", "WKR"])) + dimension + direction
    else:
        second_character = draw(st.sampled_from("OWGVLPT"))
        third_character = draw(st.sampled_from("PIF"))
        fourth_character = draw(st.sampled_from("RT"))
        return first_character + second_character + third_character + fourth_character


unit_names = st.sampled_from(
    ["SM3/DAY", "BARSA", "SM3/SM3", "FRACTION", "DAYS", "YEARS", "SM3", "SECONDS"]
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
    region_numbers: List[str]
    units: List[str]
    start_date: Date

    def to_ecl(self) -> List[Tuple[str, Any]]:
        # The restart field contains 9 strings of length 8 which
        # should contain the name of the file restarted from.
        # If shorter than 72 characters (most likely), the rest
        # are spaces. (opm manual table F.44, keyword name RESTART)
        restart = self.restart.ljust(72, " ")
        restart_list = [restart[i * 8 : i * 8 + 8] for i in range(9)]
        return [
            ("INTEHEAD", self.intehead.to_ecl()),
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
            ("NUMS    ", self.region_numbers),
            ("UNITS   ", self.units),
            ("STARTDAT", self.start_date.to_ecl()),
        ]

    def to_file(
        self, filelike, file_format: ecl_data_io.Format = ecl_data_io.Format.UNFORMATTED
    ):
        ecl_data_io.write(filelike, self.to_ecl(), file_format)


positives = st.integers(min_value=1, max_value=10000)
small_ints = st.integers(min_value=1, max_value=10)


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
    sum_keys = draw(sum_keys)
    n = len(sum_keys) + 1
    nx = draw(small_ints)
    ny = draw(small_ints)
    nz = draw(small_ints)
    keywords = ["TIME    "] + sum_keys
    units = ["DAYS    "] + draw(st.lists(unit_names, min_size=n - 1, max_size=n - 1))
    well_names = [":+:+:+:+"] + draw(st.lists(names, min_size=n - 1, max_size=n - 1))
    region_numbers = [-32676] + draw(
        st.lists(
            st.integers(min_value=0, max_value=10),
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
            ("MINISTEP", [self.mini_step]),
            ("PARAMS  ", np.array(self.params, dtype=np.float32)),
        ]


@dataclass
class SummaryStep:
    seqnum: int
    ministeps: List[SummaryMiniStep]

    def to_ecl(self):
        return [("SEQHDR  ", [self.seqnum])] + [
            i for ms in self.ministeps for i in ms.to_ecl()
        ]


@dataclass
class Unsmry:
    steps: List[SummaryStep]

    def to_ecl(self):
        return [i for step in self.steps for i in step.to_ecl()]

    def to_file(
        self, filelike, file_format: ecl_data_io.Format = ecl_data_io.Format.UNFORMATTED
    ):
        ecl_data_io.write(filelike, self.to_ecl(), file_format)


positive_floats = st.floats(min_value=0.1, allow_nan=False, allow_infinity=False)


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
