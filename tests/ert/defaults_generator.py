import datetime
from pathlib import Path

from ert.config._observations import (
    BreakthroughObservation,
    GeneralObservation,
    RFTObservation,
    SeismicObservation,
    SummaryObservation,
)
from ert.config.parsing.observations_parser import ObservationType


def create_general_observation(
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


def create_general_observation_dict(
    name: str = "general_observation",
    data: str = "RES",
    value: float | None = 1,
    error: float | None = 1,
    obs_file: str | None = None,
    restart: int | None = None,
    index_list: str | None = None,
) -> dict:
    if obs_file is not None:
        value = None
        error = None
    return {
        k: v
        for k, v in {
            "type": ObservationType.GENERAL,
            "name": name,
            "DATA": data,
            "VALUE": value,
            "ERROR": error,
            "RESTART": restart,
            "INDEX_LIST": index_list,
            "OBS_FILE": obs_file,
        }.items()
        if v is not None
    }


def create_summary_observation(
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


def create_summary_observation_dict(
    name: str = "summary_observation",
    key: str = "FOPR",
    date: str = "2020-01-01",
    value: float = 1.0,
    error: float = 0.1,
) -> dict:
    return {
        "type": ObservationType.SUMMARY,
        "name": name,
        "KEY": key,
        "DATE": date,
        "VALUE": value,
        "ERROR": error,
    }


def create_breakthrough_observation(
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


def create_breakthrough_observation_dict(
    name: str = "breakthrough_observation",
    key: str = "WWCT:OP1",
    date: datetime.datetime = datetime.datetime(2000, 3, 2, 13, 0, 0),  # noqa: DTZ001
    error: float = 10.0,
    threshold: float = 0.2,
) -> dict:
    return {
        "type": ObservationType.BREAKTHROUGH,
        "name": name,
        "KEY": key,
        "ERROR": str(error),
        "DATE": date.isoformat(),
        "THRESHOLD": threshold,
    }


def create_rft_observation(
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


def create_rft_observation_dict(
    name: str = "rft_observation",
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    zone: str | None = None,
    value: float = 150.0,
    error: float = 5.0,
) -> dict:
    return {
        "type": ObservationType.RFT,
        "name": name,
        "WELL": well,
        "DATE": date,
        "PROPERTY": prop,
        "EAST": east,
        "NORTH": north,
        "TVD": tvd,
        "VALUE": value,
        "ERROR": error,
        "ZONE": zone,
    }


def create_seismic_observation(
    name: str = "seismic_observation",
    filepath: Path = Path("obs.csv"),
    east: float = 1.0,
    north: float = 1.0,
    value: float = 1.0,
    error: float = 0.005,
    shape_id: int | None = None,
    boundary_id: int | None = None,
) -> SeismicObservation:
    return SeismicObservation(
        name=name,
        filepath=filepath,
        east=east,
        north=north,
        value=value,
        error=error,
        shape_id=shape_id,
        boundary_id=boundary_id,
    )


def create_seismic_observation_dict(
    name: str = "seismic_observation",
    csv: str = "obs.csv",
) -> dict:
    return {"type": ObservationType.SEISMIC, "name": name, "CSV": csv}
