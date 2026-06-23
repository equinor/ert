import datetime
from pathlib import Path

from ert.config._observations import (
    BreakthroughObservation,
    GeneralObservation,
    RFTObservation,
    SummaryObservation,
)


def _create_general_observation(
    name: str = "general_observation",
    data: str = "FIELD_WPR_DIFF",
    value: float = 2.0,
    error: float = 0.2,
    restart: int = 1,
    index: int = 5,
) -> GeneralObservation:
    return GeneralObservation(
        name=name,
        data=data,
        value=value,
        error=error,
        restart=restart,
        index=index,
    )


def _create_summary_observation(
    name: str = "summary_observation",
    key: str = "FOPR",
    date: str = "2020-01-01",
    value: float = 1.0,
    error: float = 0.1,
) -> SummaryObservation:
    return SummaryObservation(
        name=name,
        key=key,
        date=date,
        value=value,
        error=error,
    )


def _create_breakthrough_observation(
    name: str = "breakthrough_observation",
    key: str = "WWCT:OP1",
    date: datetime.datetime = datetime.datetime(2000, 3, 2, 13, 0, 0),  # noqa: DTZ001
    error: float = 10.0,
    threshold: float = 0.2,
) -> BreakthroughObservation:
    return BreakthroughObservation(
        name=name,
        key=key,
        date=date,
        error=error,
        threshold=threshold,
    )


def _create_rft_observation(
    name: str = "rft_observation",
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    md: float | None = 50.0,
    zone: str | None = None,
    value: float = 150.0,
    error: float = 5.0,
) -> RFTObservation:
    return RFTObservation(
        name=name,
        well=well,
        date=date,
        property=prop,
        value=value,
        error=error,
        north=north,
        east=east,
        tvd=tvd,
        md=md,
        zone=zone,
    )
