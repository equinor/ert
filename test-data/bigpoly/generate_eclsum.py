import os
import sys
from dataclasses import astuple, dataclass
from datetime import datetime, timedelta
from enum import Enum, unique
from typing import Any, List, Optional, Tuple

try:
    import resfo as ecl_data_io
except ImportError:
    import ecl_data_io

import hypothesis.strategies as st
import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.extra.numpy import from_dtype
from pydantic import PositiveInt, conint
from typing_extensions import Self

UNIT_NAMES = st.sampled_from(
    ["SM3/DAY", "BARSA", "SM3/SM3", "FRACTION", "DAYS", "YEARS", "SM3", "SECONDS"]
)

NAMES = st.text(
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
class SmspecIntehead:
    unit: UnitSystem
    simulator: Simulator

    def to_ecl(self) -> List[Any]:
        return [value.to_ecl() for value in astuple(self)]


@dataclass
class Date:
    day: conint(ge=1, le=31)
    month: conint(ge=1, le=12)
    year: conint(gt=1901, lt=2038)
    hour: conint(ge=0, lt=24)
    minutes: conint(ge=0, lt=60)
    micro_seconds: conint(ge=0, lt=60000000)

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
        return [
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
            ("LGRS    ", self.lgrs),
            ("NUMLX   ", self.numlx),
            ("NUMLY   ", self.numly),
            ("NUMLZ   ", self.numlz),
            ("LGRNAMES", self.lgr_names),
            ("NUMS    ", np.array(self.region_numbers, dtype=np.int32)),
            ("UNITS   ", self.units),
            ("STARTDAT", np.array(self.start_date.to_ecl(), dtype=np.int32)),
        ]

    def to_file(
        self, filelike, file_format: ecl_data_io.Format = ecl_data_io.Format.UNFORMATTED
    ):
        ecl_data_io.write(filelike, self.to_ecl(), file_format)


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
    sum_keys = draw(sum_keys)
    n = len(sum_keys) + 1
    nx = draw(small_ints)
    ny = draw(small_ints)
    nz = draw(small_ints)
    keywords = ["TIME    "] + sum_keys
    units = ["DAYS    "] + draw(st.lists(UNIT_NAMES, min_size=n - 1, max_size=n - 1))
    well_names = [":+:+:+:+"] + draw(st.lists(NAMES, min_size=n - 1, max_size=n - 1))
    lgrs = draw(st.lists(NAMES, min_size=n, max_size=n))
    numlx = draw(st.lists(small_ints, min_size=n, max_size=n))
    numly = draw(st.lists(small_ints, min_size=n, max_size=n))
    numlz = draw(st.lists(small_ints, min_size=n, max_size=n))
    lgr_names = list(set(lgrs))
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
            restart=NAMES,
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
        return [i for step in self.steps for i in step.to_ecl()]

    def to_file(
        self, filelike, file_format: ecl_data_io.Format = ecl_data_io.Format.UNFORMATTED
    ):
        ecl_data_io.write(filelike, self.to_ecl(), file_format)


@st.composite
def summaries(draw):
    sum_keys = [
        "AAQT",
        "AAQT",
        "BAPI",
        "BAPI",
        "BOIT",
        "BOSAT",
        "BPR",
        "BPR",
        "BTIT",
        "CTIT",
        "FAQR",
        "FAQR",
        "FGIT",
        "FGPT",
        "FGPT",
        "FLOOK",
        "FLOWI",
        "FLOWJ",
        "FLOWK",
        "FLOWK",
        "FPPT",
        "FPR",
        "FPR",
        "FPR",
        "FPR",
        "FTFT",
        "GKRR",
        "GKRZ",
        "GKRZ",
        "GKRZ-",
        "GLIR",
        "GPPR",
        "GPPT",
        "GWFT",
        "LBPFR",
        "LBTIT",
        "LCPFT",
        "LCVPT",
        "LWOIT",
        "LWOPR",
        "MAXDPR",
        "MAXDPR",
        "MAXDSG",
        "MAXDSO",
        "NAIMFRAC",
        "NAIMFRAC",
        "NALQT",
        "NBVIT",
        "NCGIR",
        "NCLIT",
        "NFGFR",
        "NFVFR",
        "NFVIT",
        "NGVFR",
        "NGWPT",
        "NLBWIT",
        "NLINSMAX",
        "NLINSMIN",
        "NLWGPR",
        "NLWLPR",
        "NRVFT",
        "NSOIR",
        "NSTPR",
        "NSVFT",
        "OKRY",
        "RANQT",
        "RCOFT",
        "RCOPT",
        "RGFR",
        "RGOPR",
        "RGPR",
        "RGVIT",
        "RLIR",
        "RNFT",
        "RNFT",
        "ROFR",
        "ROFR",
        "ROFR",
        "ROFR",
        "ROFR+",
        "ROFR-",
        "ROFT+",
        "ROFT+",
        "ROFT-",
        "ROFTG",
        "RRTFR ",
        "RSVIR",
        "RSVIT",
        "RTFR",
        "RWF T-",
        "RWFR",
        "RWLFR",
        "SALQ",
        "SALQ",
        "SFR",
        "SFR",
        "SFR",
        "SGFR",
        "SGFR",
        "SGFRF",
        "SGFRF",
        "SGFRS",
        "SGFRS",
        "SGFRS",
        "SGFT",
        "SGFT",
        "SGFTA",
        "SGFTA",
        "SOPT",
        "STEPTYPE",
        "VELGK",
        "VELOI",
        "VELOJ",
        "VELOK",
        "VELOK",
        "VELOK",
        "VELWJ",
        "WAAQR",
        "WANQT",
        "WBHP",
        "WBHP",
        "WBHPT",
        "WBP4",
        "WBP5",
        "WBP5",
        "WFGIT",
        "WGGPT",
        "WKRR",
        "WKRR-",
        "WKRX",
        "WKRX",
        "WKRX",
        "WKRZ-",
        "WKRZ-",
        "WPIO",
        "WPIO",
        "WSLIT",
        "WVIT",
        "WWCT",
        "WWCT",
        "WWCT",
    ]
    first_date = datetime.strptime("1999-1-1", "%Y-%m-%d")
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
    dates = np.arange(0.0, 50000.0)
    try:
        _ = first_date + timedelta(days=max(dates))
    except (ValueError, OverflowError):  # datetime has a max year
        print(f"Failed assumption of max_date {max(dates)}, {first_date}")
        assume(False)

    ds = sorted(dates, reverse=True)
    steps = []
    i = 0
    j = 0
    while len(ds) > 0:
        minis = []
        for _ in range(min(3, len(ds))):
            data = np.zeros(len(sum_keys) + 1)
            data[0] = ds.pop()
            minis.append(SummaryMiniStep(i, data))
            i += 1
        steps.append(SummaryStep(j, minis))
        j += 1
    return smspec, Unsmry(steps)


@settings(
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.large_base_example,
    ],
    max_examples=1,
)
@given(summaries())
def main(summary):
    smspec, unsmry = summary
    path = sys.argv[1]
    unsmry.to_file(f"{path}.UNSMRY")
    smspec.to_file(f"{path}.SMSPEC")
    os._exit(0)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
