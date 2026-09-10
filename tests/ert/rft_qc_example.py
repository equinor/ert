from __future__ import annotations

from typing import Any

from ert.config._observations import RFTObservation
from tests.ert.rft_generator import create_egrid, rft_entry

BASE_NAME = "BASE"
ZONEMAP_FILE = "zonemap.txt"
ZONEMAP = "1 zone2\n2 zone2\n3 zone2\n4 zone2\n5 zone2\n"


def _rft_obs(
    obs_name: str,
    well: str,
    east: float = 100.0,
    north: float = 100.0,
    tvd: float = 100.0,
    md: float | None = 110.0,
    value: float = 111.0,
    *,
    prop: str = "PRESSURE",
    date: str = "2000-01-01",
    zone: str | None = "zone2",
    error: float = 5.0,
) -> RFTObservation:
    return RFTObservation(
        name=obs_name,
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


# fmt: off
OBSERVATIONS = [
    #         name     well    east   north   tvd     md   value
    _rft_obs("OBS1", "WELL_A", 100.0, 100.0, 100.0, 110.0, 111.0, zone="wrong_zone"),
    _rft_obs("OBS2", "WELL_A", 100.0, 100.0, 200.0, 220.0, 222.0),
    _rft_obs("OBS3", "WELL_A", 110.0, 100.0, 290.0, 330.0, 333.0),
    _rft_obs("OBS4", "WELL_A", 100.0, 100.0, 400.0, 440.0, 444.0),
    _rft_obs("OBS5", "WELL_B", 200.0, 251.0, 300.0, 330.0, 555.0, date="2001-01-01"),
    _rft_obs("OBS6", "WELL_B", 240.0, 240.0, 420.0, 440.0, 556.0, date="2001-01-01"),
    _rft_obs("OBS7", "WELL_B", 180.0, 180.0, 480.0, 500.0, 655.0, date="2001-01-01", zone="wrong_zone"),  # ruff: ignore[line-too-long]
]

WELL_A_RFT = [
#        ijks     pressure swat   depth
    ( (1, 1, 1),   112.0,  0.4,   100.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ( (1, 1, 2),   223.0,  0.5,   200.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ( (1, 1, 3),   334.0,  0.6,   300.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ( (1, 1, 5),   556.0,  0.7,   500.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
]
WELL_B_RFT = [
#        ijks     pressure  depth
    ( (2, 2, 5),   555.0,   500.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
]
WELL_C_RFT = [
#        ijks     pressure   depth
    ( (1, 2, 1),   110.0,    100.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ( (1, 2, 2),   220.0,    200.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ( (1, 2, 3),   330.0,    300.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ( (1, 2, 4),   440.0,    400.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
]
WELL_A_TYPO_RFT = [
#        ijks     pressure  depth
    ( (1, 1, 4),   445.0,   400.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
]
# fmt: on


def _list_of_tuples_to_dict(
    keys: list[str], rows: list[tuple[Any, ...]]
) -> dict[str, Any]:
    return dict(zip(keys, zip(*rows, strict=True), strict=True))


def egrid() -> list[tuple[str, Any]]:
    """A 2x2x5 grid of 100m cells, the first cell centered on (100, 100, 100)."""
    return create_egrid(2, 2, 5, 100, 100, 100, 50, 50, 50)


def rft_file() -> list[tuple[str, Any]]:
    well_A = _list_of_tuples_to_dict(["ijks", "pressure", "swat", "depth"], WELL_A_RFT)
    well_B = _list_of_tuples_to_dict(["ijks", "pressure", "depth"], WELL_B_RFT)
    well_C = _list_of_tuples_to_dict(["ijks", "pressure", "depth"], WELL_C_RFT)
    well_A_typo = _list_of_tuples_to_dict(
        ["ijks", "pressure", "depth"], WELL_A_TYPO_RFT
    )
    return [
        *rft_entry(well_name=b"WELL_A", date=(1, 1, 2000), **well_A),
        *rft_entry(well_name=b"WELL_B", date=(1, 1, 2001), **well_B),
        *rft_entry(well_name=b"WELL_C", date=(1, 1, 2000), **well_C),
        *rft_entry(well_name=b"WELL_A_TYPO", date=(1, 1, 2000), **well_A_typo),
    ]


def observations_config() -> str:
    """The observations rendered as an OBS_CONFIG file."""
    return "\n".join(
        f"RFT_OBSERVATION {o.name} {{\n"
        f"    WELL={o.well};\n"
        f"    DATE={o.date};\n"
        f"    PROPERTY={o.property};\n"
        f"    VALUE={o.value};\n"
        f"    ERROR={o.error};\n"
        f"    EAST={o.east};\n"
        f"    NORTH={o.north};\n"
        f"    TVD={o.tvd};\n"
        f"    MD={o.md};\n"
        f"    ZONE={o.zone};\n"
        "};\n"
        for o in OBSERVATIONS
    )
