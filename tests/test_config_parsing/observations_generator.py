import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import List, Optional

import hypothesis.strategies as st
from hypothesis import assume
from pydantic import PositiveFloat


class ErrorMode(Enum):
    REL = auto()
    ABS = auto()
    RELMIN = auto()


@dataclass
class Observation(ABC):
    name: str
    error: PositiveFloat

    @property
    @abstractmethod
    def class_name(self):
        pass

    def __str__(self):
        result = f"{self.class_name} {self.name}"
        result += " { "
        for f in fields(self):
            if f.name == "name":
                continue
            val = getattr(self, f.name)
            if val is None or val == []:
                continue
            if isinstance(val, Enum):
                result += f"{f.name.upper()} = {val.name}; "
            elif isinstance(val, (float, str, int)):
                result += f"{f.name.upper()} = {val}; "
            elif isinstance(val, Observation):
                result += str(val)
            elif isinstance(val, list):
                result += f"{' '.join([str(v) for v in val])}"
            else:
                raise AssertionError
        result += " };"
        return result


@dataclass
class Segment(Observation):
    start: int
    stop: int
    error_min: PositiveFloat

    @property
    def class_name(self):
        return "SEGMENT"


@dataclass
class HistoryObservation(Observation):
    error_mode: ErrorMode
    segment: List[Segment] = field(default_factory=list)

    @property
    def class_name(self):
        return "HISTORY_OBSERVATION"

    def get_date(self, start):
        return start


@dataclass
class SummaryObservation(Observation):
    value: float
    key: str
    error_min: PositiveFloat
    error_mode: ErrorMode
    days: Optional[float] = None
    hours: Optional[float] = None
    restart: Optional[int] = None
    date: Optional[str] = None

    @property
    def class_name(self):
        return "SUMMARY_OBSERVATION"

    def get_date(self, start):
        if self.date is not None:
            return datetime.datetime.strptime(self.date, "%Y-%m-%d")
        delta = datetime.timedelta(days=0)
        if self.days is not None:
            delta += datetime.timedelta(days=self.days)
        if self.hours is not None:
            delta += datetime.timedelta(hours=self.hours)
        return start + delta


@dataclass
class GeneralObservation(Observation):
    data: str
    date: Optional[str] = None
    days: Optional[float] = None
    hours: Optional[float] = None
    restart: Optional[int] = None
    obs_file: Optional[str] = None
    value: Optional[float] = None
    index_list: Optional[List[int]] = None

    def get_date(self, start):
        if self.date is not None:
            return datetime.datetime.strptime(self.date, "%Y-%m-%d")
        delta = datetime.timedelta(0)
        if self.days is not None:
            delta += datetime.timedelta(days=self.days)
        if self.hours is not None:
            delta += datetime.timedelta(hours=self.hours)
        return start + delta

    @property
    def class_name(self):
        return "GENERAL_OBSERVATION"


names = st.text(
    min_size=1, max_size=8, alphabet=st.characters(min_codepoint=65, max_codepoint=90)
)


@st.composite
def general_observations(draw, ensemble_keys):
    kws = {}
    kws["data"] = draw(ensemble_keys)
    kws["name"] = draw(names)
    kws["error"] = draw(st.floats(min_value=0.0))
    val_type = draw(st.sampled_from(["value", "obs_file"]))
    if val_type == "value":
        kws["value"] = draw(st.floats())
    if val_type == "obs_file":
        kws["obs_file"] = draw(names)
        kws["error"] = None

    # We only generate restart=0 and no other time type
    # ("date", "days", "restart", "hours") because it
    # needs to match with a GEN_DATA report_step field
    # which we also generate so that 0 is always a
    # report_step
    kws["restart"] = 0
    return GeneralObservation(**kws)


positive_floats = st.floats(min_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def summary_observations(draw, summary_keys):
    kws = {}
    kws["name"] = draw(names)
    kws["key"] = draw(summary_keys)
    kws["error"] = draw(positive_floats)
    kws["error_min"] = draw(positive_floats)
    kws["error_mode"] = draw(st.sampled_from(ErrorMode))
    kws["value"] = draw(positive_floats)
    time_type = draw(st.sampled_from(["date", "days", "restart", "hours"]))
    if time_type == "date":
        date = draw(
            st.datetimes(
                max_value=datetime.datetime(year=2037, month=1, day=1),
                min_value=datetime.datetime(year=1999, month=1, day=2),
            )
        )
        kws["date"] = date.strftime("%Y-%m-%d")
    if time_type in ["days", "hours"]:
        kws[time_type] = draw(st.floats(min_value=1, max_value=10000))
    if time_type == "restart":
        kws[time_type] = draw(st.integers(min_value=1, max_value=10))
    return SummaryObservation(**kws)


@st.composite
def observations(draw, ensemble_keys, summary_keys):
    assume(ensemble_keys is not None or summary_keys is not None)
    observation_generators = []
    if ensemble_keys is not None:
        observation_generators.append(general_observations(ensemble_keys))
    if summary_keys is not None:
        observation_generators.append(summary_observations(summary_keys))
        observation_generators.append(
            st.builds(
                HistoryObservation,
                segment=st.lists(
                    st.builds(
                        Segment,
                        name=names,
                        start=st.integers(min_value=1, max_value=10),
                        stop=st.integers(min_value=1, max_value=10),
                    ),
                    max_size=2,
                ),
                name=summary_keys,
            ),
        )
    return draw(
        st.lists(
            st.one_of(*observation_generators),
            min_size=1,
            max_size=5,
        )
    )
