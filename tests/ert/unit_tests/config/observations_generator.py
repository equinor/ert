import datetime
from enum import Enum

import hypothesis.strategies as st
from hypothesis import assume
from pydantic import BaseModel

from ert.config._observations import (
    ErrorModes,
    GeneralObservation,
    Observation,
    SummaryObservation,
)
from ert.config.observation_config_migrations import Segment


def class_name(o: Observation):
    if isinstance(o, Segment):
        return "SEGMENT"
    if isinstance(o, GeneralObservation):
        return "GENERAL_OBSERVATION"
    if isinstance(o, SummaryObservation):
        return "SUMMARY_OBSERVATION"


def as_obs_config_content(observation: Observation) -> str:
    result = f"{class_name(observation)} {observation.name}"
    result += " { "
    for f_name in observation.model_fields:
        if f_name in {"name", "type", "index"}:
            continue

        val = getattr(observation, f_name)  # <-- get actual value

        if val is None or val == []:
            continue
        if isinstance(val, Enum):
            result += f"{f_name.upper()} = {val.name}; "
        elif isinstance(val, (float, str, int)):
            result += f"{f_name.upper()} = {val}; "
        elif isinstance(val, BaseModel):
            result += as_obs_config_content(val)  # or str(val)
        elif isinstance(val, list):
            result += f"{' '.join([as_obs_config_content(v) for v in val])}"
        else:
            raise AssertionError(f"Unexpected field type: {type(val)}")

    result += " };"
    return result


@st.composite
def general_observations(draw, ensemble_keys, std_cutoff, names):
    kws = {
        "data": draw(ensemble_keys),
        "name": draw(names),
        "error": draw(
            st.floats(min_value=std_cutoff, allow_nan=False, allow_infinity=False)
        ),
        "index": 0,
        "restart": 0,
        "value": draw(st.floats(allow_nan=False, allow_infinity=False)),
    }

    return GeneralObservation(**kws)


positive_floats = st.floats(
    min_value=0.1, max_value=1e9, allow_nan=False, allow_infinity=False
)
time_types = st.sampled_from(["date", "days", "restart", "hours"])


@st.composite
def summary_observations(draw, summary_keys, std_cutoff, names, datetimes):
    kws = {
        "name": draw(names),
        "key": draw(summary_keys),
        "error_min": draw(
            st.floats(
                min_value=std_cutoff,
                max_value=std_cutoff * 1.1,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        "error_mode": draw(st.sampled_from(ErrorModes)),
    }
    if kws["error_mode"] == ErrorModes.ABS:
        kws["error"] = draw(
            st.floats(
                min_value=std_cutoff,
                max_value=std_cutoff * 1.1,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        kws["value"] = draw(positive_floats)
    else:
        kws["error"] = draw(
            st.floats(
                min_value=0.1,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        kws["value"] = draw(
            st.floats(
                min_value=(std_cutoff / kws["error"]),
                max_value=(std_cutoff / kws["error"]) * 1.1,
                allow_nan=False,
                allow_infinity=False,
            )
        )

    _datetime = draw(datetimes)
    kws["date"] = _datetime.date().isoformat()

    return SummaryObservation(**kws)


@st.composite
def observations(draw, ensemble_keys, summary_keys, std_cutoff, start_date):
    assume(ensemble_keys is not None or summary_keys is not None)
    names = st.text(
        min_size=1,
        max_size=8,
        alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
    )
    seen = set()
    unique_names = names.filter(lambda x: x not in seen).map(lambda x: seen.add(x) or x)
    datetimes = st.datetimes(
        max_value=start_date + datetime.timedelta(days=200_000),  # ~ 300 years
        min_value=start_date + datetime.timedelta(days=1),
    )
    observation_generators = []
    if ensemble_keys is not None:
        observation_generators.append(
            general_observations(ensemble_keys, std_cutoff, unique_names)
        )
    if summary_keys is not None:
        observation_generators.append(
            summary_observations(summary_keys, std_cutoff, unique_names, datetimes),
        )

    return draw(
        st.lists(
            st.one_of(*observation_generators),
            min_size=1,
            max_size=5,
        )
    )
