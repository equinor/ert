import polars as pl
import pytest

from ert.config._shapes import PolygonShapeConfig, ShapeRegistry
from ert.config.observation_quality_control import (
    qc_rft_observations,
    qc_seismic_observations,
)
from ert.config.rft_config import RFTConfig
from tests.ert.defaults_generator import create_seismic_observation


def get_rft_observation(
    *,
    observation_key: str = "NAME",
    north: float = 1.0,
    east: float = 1.0,
    tvd: float = 1.0,
    zone: str | None = "zone1",
) -> dict[str, object]:
    return {
        "observation_key": observation_key,
        "response_key": "WELL:2000-01-01:PRESSURE",
        "well": "WELL",
        "date": "2000-01-01",
        "property": "PRESSURE",
        "value": 700.0,
        "error": 0.1,
        "north": north,
        "east": east,
        "tvd": tvd,
        "zone": zone,
    }


def get_observation_metadata(
    *,
    north: float = 1.0,
    east: float = 1.0,
    tvd: float = 1.0,
) -> dict[str, object]:
    well_connection_cell_map = {
        (1.0, 1.0, 0.5): [1, 1, 1],
        (1.0, 1.0, 1.0): [1, 1, 1],
        (1.0, 1.0, 1.5): [1, 1, 2],
        (1.0, 1.0, 2.5): [1, 1, 3],
    }

    zonemap = {
        1: ["zone1"],
        2: ["zone1", "zone2"],
        3: ["zone2"],
    }

    well_connection_cell = well_connection_cell_map[north, east, tvd]
    actual_zones = zonemap[well_connection_cell[2]]

    return {
        "north": north,
        "east": east,
        "tvd": tvd,
        "actual_zones": actual_zones,
        "well_connection_cell": well_connection_cell,
    }


def test_that_if_an_rft_observation_is_outside_the_zone_then_it_is_deactivated():
    observations = pl.DataFrame(
        [
            get_rft_observation(zone="zone2"),
        ]
    )
    observation_metadata = pl.DataFrame(
        [get_observation_metadata()], schema=RFTConfig.location_metadata_schema()
    )

    enriched_obs = RFTConfig.enrich_observations_with_metadata(
        observations, observation_metadata
    )

    qc = qc_rft_observations(enriched_obs)
    assert qc["qc_error"].to_list() == [
        "expected zone 'zone2' did not match any of the simulated zones: zone1"
    ]


@pytest.mark.parametrize(
    ("point", "expected_msg"),
    [
        pytest.param(
            (1.0, 1.0, 0.5),
            [
                None,
                "expected zone 'zone2' did not match any of the simulated zones: zone1",
            ],
            id="Point only in the zone of first observation",
        ),
        pytest.param(
            (1.0, 1.0, 1.5),
            [
                None,
                None,
            ],
            id="Point in the zone of both observations",
        ),
        pytest.param(
            (1.0, 1.0, 2.5),
            [
                "expected zone 'zone1' did not match any of the simulated zones: zone2",
                None,
            ],
            id="Point only in the zone of second observation",
        ),
    ],
)
def test_that_same_point_observations_with_different_zone_are_disabled_independently(
    point, expected_msg
):

    observations = pl.DataFrame(
        [
            get_rft_observation(
                observation_key="NAME1",
                zone="zone1",
                north=point[0],
                east=point[1],
                tvd=point[2],
            ),
            get_rft_observation(
                observation_key="NAME2",
                zone="zone2",
                north=point[0],
                east=point[1],
                tvd=point[2],
            ),
        ]
    )

    observation_metadata = pl.DataFrame(
        [get_observation_metadata(north=point[0], east=point[1], tvd=point[2])],
        schema=RFTConfig.location_metadata_schema(),
    )

    enriched_obs = RFTConfig.enrich_observations_with_metadata(
        observations, observation_metadata
    )

    qc = qc_rft_observations(enriched_obs)
    assert qc["qc_error"].to_list() == expected_msg


def test_that_observation_without_zones_are_not_disabled_by_zone_check():

    observations = pl.DataFrame(
        [
            get_rft_observation(observation_key="NAME1", zone=None),
            get_rft_observation(observation_key="NAME2", zone="zone2"),
        ]
    )

    observation_metadata = pl.DataFrame(
        [get_observation_metadata()], schema=RFTConfig.location_metadata_schema()
    )

    enriched_obs = RFTConfig.enrich_observations_with_metadata(
        observations, observation_metadata
    )

    qc = qc_rft_observations(enriched_obs)
    assert qc["qc_error"].to_list() == [
        None,
        "expected zone 'zone2' did not match any of the simulated zones: zone1",
    ]


def test_that_observations_within_boundary_stay_while_outside_are_removed():
    boundary = PolygonShapeConfig(
        vertices=[
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
        ]
    )
    shape_registry = ShapeRegistry()
    boundary_id = shape_registry.register(boundary)

    observations = pl.DataFrame(
        [
            create_seismic_observation(
                east=0.5,
                north=0.5,
                boundary_id=boundary_id,
            ),
            create_seismic_observation(
                east=1,
                north=1,
                boundary_id=boundary_id,
            ),
            create_seismic_observation(
                east=1.5,
                north=1.5,
                boundary_id=boundary_id,
            ),
            create_seismic_observation(
                east=2.5,
                north=2.5,
            ),
        ]
    )

    qc = qc_seismic_observations(observations, shape_registry)
    assert qc["east"].to_list() == [0.5, 2.5]
