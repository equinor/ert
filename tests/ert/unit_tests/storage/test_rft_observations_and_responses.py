import asyncio
import logging
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given

from ert.config import SummaryConfig
from ert.config.response_config import InvalidResponseFile
from ert.config.rft_config import RFTConfig
from ert.storage import open_storage
from ert.storage.local_ensemble import (
    _write_responses_to_storage,
)
from tests.ert.defaults_generator import _create_rft_observation


def rft_response(
    *,
    well: tuple[str, ...] = ("WELL",),
    date: tuple[date, ...] = (datetime(2000, 1, 1).date(),),  # noqa: DTZ001
    prop: tuple[str, ...] = ("SWAT",),
    depth: tuple[float, ...] = (1006.6,),
    values: tuple[float, ...] = (100.0,),
    well_connection_cell: tuple[tuple[int, int, int], ...] = ((10, 10, 10),),
    cell_center: tuple[tuple[float, float, float], ...] = ((100.0, 105.0, 1000.0),),
    cell_zones: tuple[tuple[str, ...], ...] = (("zone1",),),
) -> pl.DataFrame:
    return (
        pl.DataFrame(
            {
                "well": well,
                "property": prop,
                "time": date,
                "depth": pl.Series(depth, dtype=pl.Float32),
                "values": pl.Series(values, dtype=pl.Float32),
                "well_connection_cell": pl.Series(
                    well_connection_cell, dtype=pl.Array(pl.Int64, 3)
                ),
                "cell_center": pl.Series(cell_center, dtype=pl.Array(pl.Float32, 3)),
                "cell_zones": pl.Series(cell_zones, dtype=pl.List(pl.String)),
            }
        )
        .with_columns(pl.col("time").dt.to_string("%Y-%m-%d").alias("date"))
        .with_columns(
            pl.concat_str(
                [pl.col("well"), pl.col("date"), pl.col("property")], separator=":"
            ).alias("response_key"),
        )
        .select(RFTConfig.response_schema().keys())
        .pipe(RFTConfig._assert_schema, RFTConfig.response_schema())
    )


def rft_observation(
    *,
    name="RFT_OBS1",
    well="WELL",
    date="2000-01-01",
    prop="PRESSURE",
    value=100.0,
    error=10.0,
    east=100.0,
    north=105.0,
    tvd=1000.0,
    zone=None,
):
    return _create_rft_observation(
        name=name,
        well=well,
        date=date,
        prop=prop,
        value=value,
        error=error,
        east=east,
        north=north,
        tvd=tvd,
        zone=zone,
    )


def rft_observation1(*, zone=None):
    return rft_observation(
        prop="SWAT",
        zone=zone,
    )


def rft_observation2(*, well="WELL", zone=None):
    return rft_observation(
        name="RFT_OBS2",
        well=well,
        prop="SWAT",
        east=300.0,
        north=405.0,
        tvd=2000.0,
        zone=zone,
    )


def location_metadata(
    *,
    east=(100.0, 300.0),
    north=(105.0, 405.0),
    tvd=(1000.0, 2000.0),
    actual_zones=(("zone1",), ("zone2",)),
    well_connection_cell=((10, 10, 10), (10, 10, 10)),
) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "east": east,
            "north": north,
            "tvd": tvd,
            "actual_zones": actual_zones,
            "well_connection_cell": well_connection_cell,
            "well_connection_cell_center": pl.Series(
                [(e, n, t) for e, n, t in zip(east, north, tvd, strict=True)],
                dtype=pl.Array(pl.Float32, 3),
            ),
        },
        schema=RFTConfig.location_metadata_schema(),
    )
    RFTConfig._assert_schema(df, RFTConfig.location_metadata_schema())
    return df


def _create_rft_response_df(
    *,
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    depth: float = 8000.0,
    value: float = 148.0,
    i: int = 1,
    j: int = 2,
    k: int = 3,
    cell_center: tuple[float, float, float] = (100.0, 200.0, 25.0),
    cell_zones: tuple[str, ...] = (),
) -> pl.DataFrame:
    time = datetime.strptime(date, "%Y-%m-%d").date()  # noqa: DTZ007
    df = pl.DataFrame(
        {
            "response_key": [f"{well}:{date}:{prop}"],
            "well": [well],
            "date": [date],
            "property": [prop],
            "time": [time],
            "depth": pl.Series([depth], dtype=pl.Float32),
            "values": pl.Series([value], dtype=pl.Float32),
            "well_connection_cell": pl.Series([(i, j, k)], dtype=pl.Array(pl.Int64, 3)),
            "cell_center": pl.Series([cell_center], dtype=pl.Array(pl.Float32, 3)),
            "cell_zones": pl.Series([cell_zones], dtype=pl.List(pl.String)),
        }
    )
    return RFTConfig._assert_schema(df, RFTConfig.response_schema())


def _create_rft_location_metadata_df(
    *,
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    zone: str | None = None,
    i: int = 1,
    j: int = 2,
    k: int = 3,
) -> pl.DataFrame:
    zones = [zone] if zone is not None else []
    df = pl.DataFrame(
        {
            "east": pl.Series([east], dtype=pl.Float32),
            "north": pl.Series([north], dtype=pl.Float32),
            "tvd": pl.Series([tvd], dtype=pl.Float32),
            "actual_zones": pl.Series([zones], dtype=pl.List(pl.String)),
            "well_connection_cell": pl.Series([(i, j, k)], dtype=pl.Array(pl.Int64, 3)),
            "well_connection_cell_center": pl.Series(
                [(east, north, tvd)], dtype=pl.Array(pl.Float32, 3)
            ),
        }
    )
    return RFTConfig._assert_schema(df, RFTConfig.location_metadata_schema())


@contextmanager
def _create_rft_ensemble(
    ensemble_size,
    observations,
    *,
    with_summary=False,
    zonemap=None,
    approximate_missing_values=False,
):
    if zonemap is not None:
        zonemap_file = Path("zonemap.txt")
        zonemap_file.write_text(zonemap, encoding="utf-8")
    else:
        zonemap_file = None
    rft_config = RFTConfig(
        input_files=["DUMMY"],
        zonemap=zonemap_file,
        approximate_missing_values=approximate_missing_values,
    )
    result_config = [rft_config.model_dump(mode="json")]

    if with_summary:
        summary_config = SummaryConfig(keys=["FOPR"], input_files=["DUMMY"])
        result_config.append(summary_config.model_dump(mode="json"))

    with open_storage("storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": result_config,
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        yield storage.create_ensemble(
            experiment.id, ensemble_size=ensemble_size, name="test"
        )


def test_that_get_observations_and_responses_applies_rft_metadata(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["SWAT"]}},
        )

        obs1 = rft_observation1()
        obs2 = rft_observation2()

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response("rft", rft_response(values=(200.0,)), 0)
        ensemble.save_observation_location_metadata(location_metadata(), 0)

        ensemble.save_response("rft", rft_response(values=(300.0,)), 1)
        ensemble.save_observation_location_metadata(location_metadata(), 1)

        iens_active_index = np.array([0, 1])
        active_observations = ["RFT_OBS1"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["response_key"].to_list() == ["WELL:2000-01-01:SWAT"]
        assert obs_and_responses["index"].to_list() == ["100.0, 105.0, 1000.0, None"]
        assert obs_and_responses["observation_key"].to_list() == active_observations
        assert obs_and_responses["0"].to_list() == [200.0]
        assert obs_and_responses["1"].to_list() == [300.0]


def test_that_get_observations_and_responses_disables_observation(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["SWAT"]}},
        )

        obs1 = rft_observation1(zone="zone2")
        obs2 = rft_observation2(zone="zone3")

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response(
            "rft", rft_response(values=(200.0,), cell_zones=(("zone1",),)), 0
        )
        ensemble.save_observation_location_metadata(
            location_metadata(actual_zones=(("zone1",), ("zone3",))), 0
        )

        ensemble.save_response(
            "rft", rft_response(values=(300.0,), cell_zones=(("zone2",),)), 1
        )
        ensemble.save_observation_location_metadata(
            location_metadata(actual_zones=(("zone2",), ("zone3",))), 1
        )

        iens_active_index = np.array([0, 1])
        active_observations = ["RFT_OBS1", "RFT_OBS2"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["observation_key"].to_list() == active_observations
        msg = "expected zone 'zone2' did not match any of the simulated zones: zone1"
        assert obs_and_responses["0"].to_list() == [None, 200.0]
        assert obs_and_responses["1"].to_list() == [300.0, 300.0]
        assert obs_and_responses["qc_error_0"].to_list() == [msg, None]
        assert obs_and_responses["qc_error_1"].to_list() == [None, None]


def test_that_get_observations_and_responses_combines_error_messages(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["SWAT"]}},
        )

        obs1 = rft_observation1(zone="zone1")
        obs2 = rft_observation2(zone="zone2")

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response(
            "rft", rft_response(values=(200.0,), cell_zones=(("zone1",),)), 0
        )
        ensemble.save_observation_location_metadata(
            location_metadata(
                actual_zones=(("zone0",), ("zone2",)),
                well_connection_cell=(None, (10, 10, 10)),
            ),
            0,
        )

        ensemble.save_response(
            "rft", rft_response(values=(300.0,), cell_zones=(("zone2",),)), 1
        )
        ensemble.save_observation_location_metadata(
            location_metadata(
                actual_zones=(("zone1",), ("zone2",)),
                well_connection_cell=(None, (10, 10, 10)),
            ),
            1,
        )

        iens_active_index = np.array([0, 1])
        active_observations = ["RFT_OBS1", "RFT_OBS2"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["observation_key"].to_list() == active_observations

        msg1 = "expected zone 'zone1' did not match any of the simulated zones: zone0"
        msg2 = "did not find grid coordinate for location 100.0, 105.0, 1000.0"
        msg3 = (
            "no response matched observation data: "
            "response_key=WELL:2000-01-01:SWAT, well_connection_cell=None"
        )
        msg_real0 = f"{msg1};\n{msg2};\n{msg3}"
        msg_real1 = f"{msg2};\n{msg3}"
        assert obs_and_responses["0"].to_list() == [None, 200.0]
        assert obs_and_responses["1"].to_list() == [None, 300.0]
        assert obs_and_responses["qc_error_0"].to_list() == [msg_real0, None]
        assert obs_and_responses["qc_error_1"].to_list() == [msg_real1, None]


@pytest.mark.parametrize(
    ("obs2_kwargs", "response_kwargs", "meta_kwargs"),
    [
        pytest.param(
            {"well": "OTHER_WELL"},
            {"well": ("OTHER_WELL",)},
            {},
            id="on response key",
        ),
        pytest.param(
            {},
            {"well_connection_cell": ([11, 11, 11],)},
            {"well_connection_cell": ([10, 10, 10], [11, 11, 11])},
            id="on match key",
        ),
    ],
)
def test_that_get_observations_and_responses_adds_qc_error_on_rft_mismatch(
    tmp_path, obs2_kwargs, response_kwargs, meta_kwargs
):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["SWAT"]}},
        )

        obs1 = rft_observation1()
        obs2 = rft_observation2(**obs2_kwargs)

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response("rft", rft_response(**response_kwargs), 0)
        ensemble.save_response("rft", rft_response(**response_kwargs), 1)

        ensemble.save_observation_location_metadata(location_metadata(**meta_kwargs), 0)
        ensemble.save_observation_location_metadata(location_metadata(**meta_kwargs), 1)

        iens_active_index = np.array([0, 1])
        active_observations = ["RFT_OBS1", "RFT_OBS2"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["observation_key"].to_list() == active_observations

        msg = (
            "no response matched observation data: "
            "response_key=WELL:2000-01-01:SWAT, well_connection_cell=[10, 10, 10]"
        )
        np.testing.assert_equal(obs_and_responses["0"].to_list(), [None, 100.0])
        np.testing.assert_equal(obs_and_responses["1"].to_list(), [None, 100.0])
        assert obs_and_responses["qc_error_0"].to_list() == [msg, None]
        assert obs_and_responses["qc_error_1"].to_list() == [msg, None]


@pytest.mark.parametrize(
    (
        "approximate_missing_values",
        "missing_required_columns",
        "expected_response_values",
    ),
    [
        pytest.param(True, False, [110.0, 310.0], id="interpolate enabled"),
        pytest.param(False, False, [None, None], id="interpolate disabled"),
        pytest.param(True, True, [None, None], id="missing required columns"),
    ],
)
def test_that_get_observations_and_responses_interpolates_rft_values(
    tmp_path,
    approximate_missing_values,
    missing_required_columns,
    expected_response_values,
):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["PRESSURE"]}},
            approximate_missing_values=approximate_missing_values,
        )

        obs1 = rft_observation(name="RFT_OBS1", value=100.0, tvd=1000.0, zone="zone1")
        obs2 = rft_observation(name="RFT_OBS2", value=200.0, tvd=2000.0, zone="zone1")

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        date = datetime(2000, 1, 1).date()  # noqa: DTZ001

        # If the response is missing the required columns for interpolation,
        # we should skip interpolation and return None, even if interpolation is
        # enabled. The test drops the columns to simulate a legacy response without
        # these columns.
        drop_columns = ["cell_center", "cell_zones"] if missing_required_columns else []
        ensemble.save_response(
            "rft",
            rft_response(
                well=("WELL", "WELL"),
                date=(date, date),
                prop=("PRESSURE", "PRESSURE"),
                depth=(500.0, 1500.0),
                values=(50.0, 150.0),
                well_connection_cell=((10, 10, 10), (10, 10, 12)),
                cell_center=((100.0, 105.0, 700.0), (100.0, 105.0, 1200.0)),
                cell_zones=(("zone1",), ("zone1",)),
            ).drop(drop_columns),
            0,
        )
        ensemble.save_observation_location_metadata(
            location_metadata(
                east=(100.0, 100.0),
                north=(105.0, 105.0),
                tvd=(1000.0, 2000.0),
                actual_zones=(("zone1",), ("zone1",)),
                well_connection_cell=((10, 10, 11), (10, 10, 13)),
            ),
            0,
        )

        iens_active_index = np.array([0])
        active_observations = ["RFT_OBS1", "RFT_OBS2"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["response_key"].to_list() == [
            "WELL:2000-01-01:PRESSURE",
            "WELL:2000-01-01:PRESSURE",
        ]
        np.testing.assert_array_equal(
            obs_and_responses["0"].to_list(), expected_response_values
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_returns_joined_data():
    with _create_rft_ensemble(
        1, [_create_rft_observation(zone="Z1")], zonemap="1 Z1"
    ) as ensemble:
        realization = 0
        ensemble.save_response(
            "rft",
            pl.concat(
                [
                    _create_rft_response_df(prop="PRESSURE", value=148.0),
                    _create_rft_response_df(prop="SGAS", value=0.1),
                    _create_rft_response_df(prop="SWAT", value=0.2),
                ]
            ),
            realization,
        )
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(zone="Z1"),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses()

        assert {k: v.to_list() for k, v in result.to_dict().items()} == {
            "order": [0],
            "utm_x": [100.0],
            "utm_y": [200.0],
            "measured_depth": [50.0],
            "true_vertical_depth": [25.0],
            "zone": ["Z1"],
            "pressure": [148.0],
            "swat": [pytest.approx(0.2)],
            "sgas": [pytest.approx(0.1)],
            "soil": [pytest.approx(0.7)],  # soil is computed as 1 - sgas - swat
            "valid_zone": [True],
            "is_active": [True],
            "i": [1],
            "j": [2],
            "k": [3],
            "well": ["WELL1"],
            "time": ["2020-01-01"],
            "realization": [0],
            "report_step": [0],
            "observed": [150.0],
            "error": [5.0],
        }


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("approximate_missing_values", "expected_approximated_values"),
    [
        (True, 165.0),
        (False, None),
    ],
)
def test_that_get_rft_observations_and_responses_includes_approximated_values_when_enabled(  # noqa: E501
    approximate_missing_values, expected_approximated_values
):
    observation = _create_rft_observation(zone="zone1")
    with _create_rft_ensemble(
        1, [observation], approximate_missing_values=approximate_missing_values
    ) as ensemble:
        realization = 0
        # Save two responses that does not match the observation, but can be used to
        # approximate a response at the observation location.
        ensemble.save_response(
            "rft",
            pl.concat(
                [
                    _create_rft_response_df(
                        i=0,
                        j=0,
                        k=1,
                        cell_center=(100.0, 200.0, 20.0),
                        cell_zones=("zone1",),
                        value=160.0,
                    ),
                    _create_rft_response_df(
                        i=0,
                        j=0,
                        k=3,
                        cell_center=(100.0, 200.0, 30.0),
                        cell_zones=("zone1",),
                        value=170.0,
                    ),
                ]
            ),
            realization,
        )
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(i=0, j=0, k=2, zone="zone1"),
            realization,
        )
        result = ensemble.get_rft_observations_and_responses()

        # The approximated response value is in the result dataframe:
        assert result["pressure"].to_list() == [
            pytest.approx(expected_approximated_values)
        ]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_is_active_based_on_matching_pressure_response():
    """Test that is_active is True when there is a matching PRESSURE response,
    and False if not
    """
    observations = [
        _create_rft_observation(),
        _create_rft_observation(
            name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
    ]

    with _create_rft_ensemble(1, observations) as ensemble:
        realization = 0
        ensemble.save_response(
            "rft", pl.concat([_create_rft_response_df()]), realization
        )
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(), realization
        )

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["is_active"].to_list() == [
            True,
            False,
        ]  # tvd=25 has pressure response, tvd=30 does not


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_sets_valid_zone_with_null_equality():
    observations = [
        _create_rft_observation(zone="Z1"),
        _create_rft_observation(
            name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
        _create_rft_observation(
            name="obs3",
            tvd=35.0,
            md=70.0,
            zone="Z2",
            value=170.0,
        ),
    ]

    realization = 0
    with _create_rft_ensemble(1, observations, zonemap="1 Z1\n2 Z2\n") as ensemble:
        ensemble.save_response(
            "rft",
            _create_rft_response_df(),
            realization,
        )
        ensemble.save_observation_location_metadata(
            pl.concat(
                [
                    _create_rft_location_metadata_df(zone="Z1", tvd=25.0),
                    _create_rft_location_metadata_df(zone=None, tvd=30.0),
                    _create_rft_location_metadata_df(zone="Z1", tvd=35.0),
                ]
            ),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["valid_zone"][0] is True  # Z1 == Z1
        assert result["valid_zone"][1] is True  # None == None
        assert result["valid_zone"][2] is False  # Z2 != Z1


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_order_is_row_index_within_well():
    observations = [
        _create_rft_observation(name="obs1", tvd=25.0, md=50.0),
        _create_rft_observation(name="obs2", tvd=30.0, md=60.0),
        _create_rft_observation(name="obs3", tvd=35.0, md=70.0),
        _create_rft_observation(well="WELL2", name="obs4", tvd=40.0, md=80.0),
        _create_rft_observation(well="WELL2", name="obs5", tvd=45.0, md=90.0),
    ]

    with _create_rft_ensemble(1, observations) as ensemble:
        realization = 0
        ensemble.save_response(
            "rft", pl.concat([_create_rft_response_df()]), realization
        )
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses().sort(
            ["well", "true_vertical_depth"]
        )

        assert result["well"].to_list() == ["WELL1", "WELL1", "WELL1", "WELL2", "WELL2"]
        assert result["order"].to_list() == [0, 1, 2, 0, 1]


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.slow
@given(
    response_values=st.lists(
        st.floats(allow_infinity=False, allow_nan=False, width=32), min_size=1
    ),
    prop=st.sampled_from(["PRESSURE", "SWAT", "SGAS"]),
)
def test_that_get_rft_observations_and_responses_handles_multiple_realizations(
    response_values: list[float], prop: str
):
    with _create_rft_ensemble(
        len(response_values), [_create_rft_observation()]
    ) as ensemble:
        for i, r in enumerate(response_values):
            ensemble.save_response(
                "rft", _create_rft_response_df(value=r, prop=prop), i
            )
            ensemble.save_observation_location_metadata(
                _create_rft_location_metadata_df(),
                i,
            )

        result = ensemble.get_rft_observations_and_responses().sort("realization")

        assert result.shape[0] == len(response_values)
        assert result["realization"].to_list() == list(range(len(response_values)))

        for i, r in enumerate(response_values):
            assert result[prop.lower()][i] == pytest.approx(r)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_raises_error_for_no_observations():
    with (
        _create_rft_ensemble(1, []) as ensemble,
        pytest.raises(ValueError, match="No RFT observations found"),
    ):
        ensemble.get_rft_observations_and_responses()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_raises_error_when_response_not_saved():
    with (
        _create_rft_ensemble(1, [_create_rft_observation()]) as ensemble,
        pytest.raises(KeyError, match="No response for key rft"),
    ):
        ensemble.get_rft_observations_and_responses()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_raises_error_when_location_metadata_not_saved():  # noqa: E501
    with _create_rft_ensemble(1, [_create_rft_observation()]) as ensemble:
        ensemble.save_response("rft", _create_rft_response_df(), 0)
        with pytest.raises(FileNotFoundError, match="observation_location_metadata"):
            ensemble.get_rft_observations_and_responses()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_adds_missing_saturation_columns():
    realization = 0
    with _create_rft_ensemble(1, [_create_rft_observation()]) as ensemble:
        # Save a response that does not contain saturations
        ensemble.save_response("rft", _create_rft_response_df(), realization)
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses()

        for saturation in ["sgas", "swat", "soil"]:
            assert saturation in result.columns
            assert result[saturation].to_list() == [None]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_maps_report_step_from_summary_times(
    tmp_path,
):
    observations = [
        _create_rft_observation(date="2020-01-15"),
        _create_rft_observation(
            date="2020-02-15",
            name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
    ]

    rft_responses = pl.concat(
        [_create_rft_response_df(value=148.0), _create_rft_response_df(value=158.0)]
    )

    summary_responses = pl.DataFrame(
        {
            "response_key": ["FOPR"] * 3,
            "time": pl.Series(
                [datetime(2020, 1, 1), datetime(2020, 1, 15), datetime(2020, 2, 15)]  # noqa: DTZ001
            ).dt.cast_time_unit("ms"),
            "values": pl.Series([100.0, 200.0, 300.0], dtype=pl.Float32),
        }
    )

    with _create_rft_ensemble(1, observations, with_summary=True) as ensemble:
        realization = 0
        ensemble.save_response("rft", rft_responses, realization)
        ensemble.save_response("summary", summary_responses, realization)
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses().sort("time")

        assert "report_step" in result.columns
        assert result["report_step"].to_list() == [1, 2]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_rft_artifacts_are_saved_atomically(caplog):
    caplog.set_level(logging.WARNING)

    def read_from_file_mock(run_path, iens, iter_) -> pl.DataFrame:
        if iens == 1:
            raise InvalidResponseFile("Mock error for realization 1")
        if iens == 3:
            raise Exception("Mock error for realization 3")
        return _create_rft_response_df()

    def location_metadata_mock(run_path, iens, iter_, observations) -> pl.DataFrame:
        if iens == 2:
            raise InvalidResponseFile("Mock error for realization 2")
        if iens == 4:
            raise Exception("Mock error for realization 4")
        return _create_rft_location_metadata_df()

    async def run_test():
        with (
            patch.object(RFTConfig, "read_from_file", side_effect=read_from_file_mock),
            patch.object(
                RFTConfig,
                "obtain_location_metadata",
                side_effect=location_metadata_mock,
            ),
        ):
            num_realizations = 5
            observations = [
                _create_rft_observation(),
            ]

            with _create_rft_ensemble(num_realizations, observations) as ensemble:
                results = []
                for realization in range(num_realizations):
                    result = await _write_responses_to_storage(
                        "", realization, ensemble
                    )
                    results.append(result)

                assert results[0].successful
                for realization in range(1, num_realizations):
                    assert not results[realization].successful

                logs = [
                    (r.levelno, " ".join(r.getMessage().split()[:5]))
                    for r in caplog.records
                ]
                assert logs == [
                    (logging.WARNING, "Failed to read response from"),
                    (logging.WARNING, "Failed to write observation metadata"),
                    (logging.ERROR, "Unexpected exception while reading from"),
                    (logging.ERROR, "Unexpected exception while writing RFT"),
                ]

                ensemble.load_responses("rft", (0,))
                for realization in range(1, num_realizations):
                    with pytest.raises(KeyError):
                        ensemble.load_responses("rft", (realization,))

                ensemble.load_observation_location_metadata(0)
                for realization in range(1, num_realizations):
                    with pytest.raises(FileNotFoundError):
                        ensemble.load_observation_location_metadata(realization)

                assert ensemble.get_realization_list_with_responses() == [0]

    asyncio.run(run_test())


@pytest.mark.usefixtures("use_tmpdir")
def test_that_location_metadata_file_is_not_created_when_no_observations():
    def read_from_file_mock(run_path, iens, iter_):
        return _create_rft_response_df()

    async def run_test():
        with (
            patch.object(RFTConfig, "read_from_file", side_effect=read_from_file_mock),
        ):
            num_realizations = 1
            observations = []

            with _create_rft_ensemble(num_realizations, observations) as ensemble:
                await _write_responses_to_storage("", 0, ensemble)

                ensemble.load_responses("rft", (0,))
                with pytest.raises(FileNotFoundError):
                    ensemble.load_observation_location_metadata(0)
                assert ensemble.get_realization_list_with_responses() == [0]

    asyncio.run(run_test())
