import random
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import polars as pl
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from polars.testing import assert_frame_equal
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QAbstractItemView

from ert.config import ErtConfig
from ert.config._create_observation_dataframes import _rft_observation_schema
from ert.config._observations import RFTObservation
from ert.config.response_config import InvalidResponseFile
from ert.config.rft_config import RFTConfig
from ert.gui.tools.manage_experiments.rft_qc_widget import (
    GREY,
    CurrentRealization,
    DisplayPoints,
    PlotControlPanel,
    RftPlot,
    RftQcWidget,
    _deduplicate_points_per_coordinate_and_status,
    _PointStatus,
)
from ert.storage import (
    open_storage,
)
from ert.storage.local_ensemble import (
    LocalEnsemble,
    _write_observation_metadata,
)
from tests.ert.rft_generator import cell_start, create_egrid

float_arr = partial(np.array, dtype=np.float32)


@contextmanager
def _create_rft_ensemble(
    ensemble_size,
    observations,
    *,
    data_to_read=None,
    zonemap=None,
    approximate_missing_values=False,
):
    rft_config = RFTConfig(
        input_files=["BASE"],
        data_to_read=data_to_read if data_to_read is not None else {},
        zonemap=zonemap,
        approximate_missing_values=approximate_missing_values,
    )
    data_to_read = rft_config.data_to_read
    for rft_observation in observations:
        if rft_observation.well not in data_to_read:
            rft_config.data_to_read[rft_observation.well] = {}

        well_dict = data_to_read[rft_observation.well]
        if rft_observation.date not in well_dict:
            well_dict[rft_observation.date] = []

        property_list = well_dict[rft_observation.date]
        if rft_observation.property not in property_list:
            property_list.append(rft_observation.property)

    result_config = [rft_config.model_dump(mode="json")]

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


def _rft_entry(
    well_name: bytes,
    date: tuple[int, int, int],
    ijks: tuple[tuple[int, int, int]],
    **kwargs,
):

    return [
        *cell_start(well_name=well_name, date=date, ijks=ijks),
        *[(k.ljust(8).upper(), float_arr(v)) for k, v in kwargs.items()],
    ]


def _list_of_tuples_to_dict(keys: list[str], tuples: list[tuple]) -> dict[str, Any]:
    return dict(zip(keys, zip(*tuples, strict=True), strict=True))


@contextmanager
def _loaded_multi_well_rft_qc_widget(qtbot, mocked_files, mock_resfo_file):
    # fmt: off
    observations = [
        #         name     well    east   north   tvd     md   value
        _rft_obs("OBS1", "WELL_A", 100.0, 100.0, 100.0, 110.0, 111.0, zone="wrong_zone"),  # ruff: ignore[line-too-long]
        _rft_obs("OBS2", "WELL_A", 100.0, 100.0, 200.0, 220.0, 222.0),
        _rft_obs("OBS3", "WELL_A", 110.0, 100.0, 290.0, 330.0, 333.0),
        _rft_obs("OBS4", "WELL_A", 100.0, 100.0, 400.0, 440.0, 444.0),
        _rft_obs("OBS5", "WELL_B", 200.0, 251.0, 300.0, 330.0, 555.0, date="2001-01-01"),  # ruff: ignore[line-too-long]
        _rft_obs("OBS6", "WELL_B", 240.0, 240.0, 420.0, 440.0, 556.0, date="2001-01-01"),  # ruff: ignore[line-too-long]
        _rft_obs("OBS7", "WELL_B", 180.0, 180.0, 480.0, 500.0, 655.0, date="2001-01-01", zone="wrong_zone"),  # ruff: ignore[line-too-long]
    ]

    well_A_rft = [
    #        ijks     pressure swat   depth
        ( (1, 1, 1),   112.0,  0.4,   100.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( (1, 1, 2),   223.0,  0.5,   200.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( (1, 1, 3),   334.0,  0.6,   300.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( (1, 1, 5),   556.0,  0.7,   500.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ]
    well_B_rft = [
    #        ijks     pressure  depth
        ( (2, 2, 5),   555.0,   500.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ]
    well_C_rft = [
    #        ijks     pressure   depth
        ( (1, 2, 1),   110.0,    100.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( (1, 2, 2),   220.0,    200.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( (1, 2, 3),   330.0,    300.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( (1, 2, 4),   440.0,    400.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ]
    well_A_typo_rft = [
    #        ijks     pressure  depth
        ( (1, 1, 4),   445.0,   400.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ]
    # fmt: on

    well_A = _list_of_tuples_to_dict(["ijks", "pressure", "swat", "depth"], well_A_rft)
    well_B = _list_of_tuples_to_dict(["ijks", "pressure", "depth"], well_B_rft)
    well_C = _list_of_tuples_to_dict(["ijks", "pressure", "depth"], well_C_rft)
    well_A_typo = _list_of_tuples_to_dict(
        ["ijks", "pressure", "depth"], well_A_typo_rft
    )
    BASE_PATH = "path/does/not/exist"
    mocked_files[f"{BASE_PATH}/zonemap.txt"] = (
        "1 zone2\n2 zone2\n3 zone2\n4 zone2\n5 zone2\n"
    )
    mock_resfo_file(
        f"{BASE_PATH}/BASE.EGRID",
        create_egrid(2, 2, 5, 100, 100, 100, 50, 50, 50),
    )
    mock_resfo_file(
        f"{BASE_PATH}/BASE.RFT",
        [
            *_rft_entry(date=(1, 1, 2000), well_name=b"WELL_A", **well_A),
            *_rft_entry(date=(1, 1, 2001), well_name=b"WELL_B", **well_B),
            *_rft_entry(date=(1, 1, 2000), well_name=b"WELL_C", **well_C),
            *_rft_entry(date=(1, 1, 2000), well_name=b"WELL_A_TYPO", **well_A_typo),
        ],
    )

    realization = 0
    with _create_rft_ensemble(
        1,
        observations,
        data_to_read={"WELL_A": {"*": ["*"]}},
        zonemap="zonemap.txt",
        approximate_missing_values=True,
    ) as ensemble:
        rft_config = cast(RFTConfig, ensemble.experiment.response_configuration["rft"])
        _write_observation_metadata(BASE_PATH, realization, ensemble)
        rft_responses_df = rft_config.read_from_file(
            BASE_PATH, realization, ensemble.iteration
        )
        ensemble.save_response(rft_config.type, rft_responses_df, realization)
        widget = RftQcWidget()
        widget._get_runpath = lambda ensemble, realization: BASE_PATH
        widget.update_realization(ensemble, realization)
        yield widget


def _display_all_points(widget) -> None:
    widget._on_toggle_file_rft(True)
    for list_widget in (
        widget._filter_panel._date_list,
        widget._filter_panel._property_list,
    ):
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        list_widget.selectAll()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_rft_qc_widget_loads_and_displays_observations_responses_and_file_rft(
    qtbot, mocked_files, mock_resfo_file
):
    def expected_obs() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "response_key": ["WELL_A:2000-01-01:PRESSURE"] * 4
                + ["WELL_B:2001-01-01:PRESSURE"] * 3,
                "well": ["WELL_A"] * 4 + ["WELL_B"] * 3,
                "date": ["2000-01-01"] * 4 + ["2001-01-01"] * 3,
                "property": ["PRESSURE"] * 7,
                "zone": ["wrong_zone"] + ["zone2"] * 5 + ["wrong_zone"],
                "observations": [111.0, 222.0, 333.0, 444.0, 555.0, 556.0, 655.0],
                "std": [5.0] * 7,
                "well_connection_cell": [
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 1, 3],
                    [1, 1, 4],
                    None,
                    [2, 2, 4],
                    [2, 2, 5],
                ],
                "well_connection_cell_center": [
                    [100.0, 100.0, 100.0],
                    [100.0, 100.0, 200.0],
                    [100.0, 100.0, 300.0],
                    [100.0, 100.0, 400.0],
                    None,
                    [200.0, 200.0, 400.0],
                    [200.0, 200.0, 500.0],
                ],
                "east": [100.0, 100.0, 110.0, 100.0, 200.0, 240.0, 180.0],
                "north": [100.0, 100.0, 100.0, 100.0, 251.0, 240.0, 180.0],
                "tvd": [100.0, 200.0, 290.0, 400.0, 300.0, 420.0, 480.0],
                "actual_zones": [["zone2"]] * 4 + [[]] + [["zone2"]] * 2,
                "qc_error": [
                    (
                        "expected zone 'wrong_zone' did not match any of the simulated "
                        "zones: zone2"
                    )
                ]
                + [None] * 3
                + [
                    (
                        "expected zone 'zone2' did not match any of the simulated zones: ;\n"  # ruff: ignore[line-too-long]
                        "did not find grid coordinate for location 200.0, 251.0, 300.0"
                    )
                ]
                + [None]
                + [
                    (
                        "expected zone 'wrong_zone' did not match any of the simulated "
                        "zones: zone2"
                    )
                ],
                "status": [
                    _PointStatus.INVALID_ZONE,
                    _PointStatus.MATCHED,
                    _PointStatus.MATCHED,
                    _PointStatus.MATCHED,
                    _PointStatus.NOT_IN_GRID,
                    _PointStatus.NO_RESPONSE,
                    _PointStatus.INVALID_ZONE,
                ],
                "values": [
                    112.0,
                    223.0,
                    334.0,
                    445.0,
                    None,
                    None,
                    555.0,
                ],
            },
            schema=RftQcWidget._required_obs_subschema(),
        )

    def expected_responses() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "response_key": ["WELL_A:2000-01-01:PRESSURE"] * 4
                + ["WELL_A:2000-01-01:SWAT"] * 4
                + ["WELL_B:2001-01-01:PRESSURE"]
                + ["WELL_A:2000-01-01:PRESSURE"],
                "well": ["WELL_A"] * 8 + ["WELL_B"] + ["WELL_A"],
                "date": ["2000-01-01"] * 8 + ["2001-01-01"] + ["2000-01-01"],
                "property": ["PRESSURE"] * 4 + ["SWAT"] * 4 + ["PRESSURE"] * 2,
                "values": [
                    112.0,
                    223.0,
                    334.0,
                    556.0,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    555.0,
                    445.0,
                ],
                "well_connection_cell": [
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 1, 3],
                    [1, 1, 5],
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 1, 3],
                    [1, 1, 5],
                    [2, 2, 5],
                    [1, 1, 4],
                ],
                "well_connection_cell_center": [
                    [100.0, 100.0, 100.0],
                    [100.0, 100.0, 200.0],
                    [100.0, 100.0, 300.0],
                    [100.0, 100.0, 500.0],
                    [100.0, 100.0, 100.0],
                    [100.0, 100.0, 200.0],
                    [100.0, 100.0, 300.0],
                    [100.0, 100.0, 500.0],
                    [200.0, 200.0, 500.0],
                    [100.0, 100.0, 400.0],
                ],
                "east": [100.0] * 8 + [200.0, 100.0],
                "north": [100.0] * 8 + [200.0, 100.0],
                "tvd": [100.0, 200.0, 300.0, 500.0] * 2 + [500.0, 400.0],
                "depth": [100.0, 200.0, 300.0, 500.0] * 2 + [500.0, 400.0],
                "cell_zones": [["zone2"]] * 10,
                "status": [_PointStatus.RESPONSE] * 9 + [_PointStatus.APPROXIMATED],
            },
            schema=RftQcWidget._required_response_subschema(),
        )

    def expected_file_responses() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "well": ["WELL_A"] * 8 + ["WELL_B"] + ["WELL_C"] * 4 + ["WELL_A_TYPO"],
                "date": ["2000-01-01"] * 8
                + ["2001-01-01"]
                + ["2000-01-01"] * 4
                + ["2000-01-01"],
                "property": ["PRESSURE"] * 4
                + ["SWAT"] * 4
                + ["PRESSURE"]
                + ["PRESSURE"] * 4
                + ["PRESSURE"],
                "values": [
                    112.0,
                    223.0,
                    334.0,
                    556.0,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    555.0,
                    110.0,
                    220.0,
                    330.0,
                    440.0,
                    445.0,
                ],
                "well_connection_cell": [
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 1, 3],
                    [1, 1, 5],
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 1, 3],
                    [1, 1, 5],
                    [2, 2, 5],
                    [1, 2, 1],
                    [1, 2, 2],
                    [1, 2, 3],
                    [1, 2, 4],
                    [1, 1, 4],
                ],
                "well_connection_cell_center": [
                    [100.0, 100.0, 100.0],
                    [100.0, 100.0, 200.0],
                    [100.0, 100.0, 300.0],
                    [100.0, 100.0, 500.0],
                    [100.0, 100.0, 100.0],
                    [100.0, 100.0, 200.0],
                    [100.0, 100.0, 300.0],
                    [100.0, 100.0, 500.0],
                    [200.0, 200.0, 500.0],
                    [100.0, 200.0, 100.0],
                    [100.0, 200.0, 200.0],
                    [100.0, 200.0, 300.0],
                    [100.0, 200.0, 400.0],
                    [100.0, 100.0, 400.0],
                ],
                "east": [100.0] * 8 + [200.0] + [100.0] * 5,
                "north": [100.0] * 8 + [200.0] + [200.0] * 4 + [100.0],
                "tvd": [100.0, 200.0, 300.0, 500.0] * 2
                + [500.0, 100.0, 200.0, 300.0, 400.0, 400.0],
                "depth": [100.0, 200.0, 300.0, 500.0] * 2
                + [500.0, 100.0, 200.0, 300.0, 400.0, 400.0],
                "cell_zones": [["zone2"]] * 14,
                "status": [_PointStatus.FILE_RFT] * 14,
            },
            schema=RftQcWidget._required_file_response_subschema(),
        )

    def expected_displayed_points_ijk() -> pl.DataFrame:
        # fmt: off
        displayed_points = [
        # well_connection_cell   east   north    tvd           status
            (   (1, 1, 2),      100.0,  100.0,  200.0,    _PointStatus.MATCHED),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 3),      110.0,  100.0,  290.0,    _PointStatus.MATCHED),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 4),      100.0,  100.0,  400.0,    _PointStatus.MATCHED),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 1),      100.0,  100.0,  100.0,    _PointStatus.INVALID_ZONE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (2, 2, 5),      180.0,  180.0,  480.0,    _PointStatus.INVALID_ZONE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (2, 2, 4),      240.0,  240.0,  420.0,    _PointStatus.NO_RESPONSE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 1),      100.0,  100.0,  100.0,    _PointStatus.RESPONSE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 2),      100.0,  100.0,  200.0,    _PointStatus.RESPONSE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 3),      100.0,  100.0,  300.0,    _PointStatus.RESPONSE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 5),      100.0,  100.0,  500.0,    _PointStatus.RESPONSE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (2, 2, 5),      200.0,  200.0,  500.0,    _PointStatus.RESPONSE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 4),      100.0,  100.0,  400.0,    _PointStatus.APPROXIMATED),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 1),      100.0,  100.0,  100.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 2),      100.0,  100.0,  200.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 3),      100.0,  100.0,  300.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 5),      100.0,  100.0,  500.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (2, 2, 5),      200.0,  200.0,  500.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 2, 1),      100.0,  200.0,  100.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 2, 2),      100.0,  200.0,  200.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 2, 3),      100.0,  200.0,  300.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 2, 4),      100.0,  200.0,  400.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (1, 1, 4),      100.0,  100.0,  400.0,    _PointStatus.FILE_RFT),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ]
        # fmt: on
        return pl.DataFrame(
            displayed_points,
            schema={
                "well_connection_cell": pl.Array(pl.Int64, 3),
                "east": pl.Float32,
                "north": pl.Float32,
                "tvd": pl.Float32,
                "status": pl.String,
            },
            orient="row",
        )

    def expected_displayed_points_utm() -> pl.DataFrame:
        # fmt: off
        overlay_and_not_in_grid_points = [
        # well_connection_cell   east   north    tvd           status
            (        None,      200.0,  251.0, 300.0,    _PointStatus.NOT_IN_GRID),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma, multiple-spaces-before-keyword]
            (   (1, 1, 3),      100.0,  100.0, 300.0,    _PointStatus.MATCHED),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (2, 2, 5),      200.0,  200.0, 500.0,    _PointStatus.INVALID_ZONE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
            (   (2, 2, 4),      200.0,  200.0, 400.0,    _PointStatus.NO_RESPONSE),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ]
        # fmt: on
        return pl.concat(
            [
                expected_displayed_points_ijk(),
                pl.DataFrame(
                    overlay_and_not_in_grid_points,
                    schema={
                        "well_connection_cell": pl.Array(pl.Int64, 3),
                        "east": pl.Float32,
                        "north": pl.Float32,
                        "tvd": pl.Float32,
                        "status": pl.String,
                    },
                    orient="row",
                ),
            ]
        )

    with _loaded_multi_well_rft_qc_widget(
        qtbot, mocked_files, mock_resfo_file
    ) as widget:
        expected_obs_df = expected_obs()
        assert_frame_equal(
            widget._observations.select(expected_obs_df.columns), expected_obs_df
        )
        expected_responses_df = expected_responses()
        assert_frame_equal(
            widget._responses.select(expected_responses_df.columns),
            expected_responses_df,
        )

        # RFT file is not loaded by default, but can be enabled by clicking checkbox:
        assert not widget._load_rft_file
        assert widget._file_responses.is_empty()
        assert widget._load_rft_file_toggle.isEnabled()
        widget._load_rft_file_toggle.setChecked(True)

        expected_file_responses_df = expected_file_responses()
        assert_frame_equal(
            widget._file_responses.select(expected_file_responses_df.columns),
            expected_file_responses_df,
        )

        def _filter_values(list_widget) -> list[str]:
            return [
                list_widget.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(list_widget.count())
            ]

        # Filters should be populated based on observations, responses
        # and RFT file content, and sorted.
        assert _filter_values(widget._filter_panel._well_list) == [
            "WELL_A",
            "WELL_A_TYPO",
            "WELL_B",
            "WELL_C",
        ]
        assert _filter_values(widget._filter_panel._date_list) == [
            "2000-01-01",
            "2001-01-01",
        ]
        assert _filter_values(widget._filter_panel._property_list) == [
            "PRESSURE",
            "SWAT",
        ]
        assert _filter_values(widget._filter_panel._status_list) == sorted(
            status.value for status in _PointStatus
        )

        # Date and Property default to strict single-select; switch them to
        # multiselect and select all so the full point set is displayed.
        for list_widget in (
            widget._filter_panel._date_list,
            widget._filter_panel._property_list,
        ):
            list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            list_widget.selectAll()

        # Points are displayed in IJK coordinates by default:
        assert_frame_equal(
            widget._plot._display_points._df,
            expected_displayed_points_ijk(),
        )

        cells = widget._plot._display_points._df["well_connection_cell"]
        xs, ys, zs = widget._plot._point_artist._offsets3d
        np.testing.assert_array_equal(xs, cells.arr.get(0).to_list())
        np.testing.assert_array_equal(ys, cells.arr.get(1).to_list())
        np.testing.assert_array_equal(zs, cells.arr.get(2).to_list())

        # Points will be displayed in UTM coordinates when the toggle is checked:
        widget._filter_panel._toggle_utm_coords.setChecked(True)
        assert_frame_equal(
            widget._plot._display_points._df,
            expected_displayed_points_utm(),
            check_row_order=False,
        )
        xs, ys, zs = widget._plot._point_artist._offsets3d
        np.testing.assert_array_equal(
            xs, widget._plot._display_points._df["east"].to_list()
        )
        np.testing.assert_array_equal(
            ys, widget._plot._display_points._df["north"].to_list()
        )
        np.testing.assert_array_equal(
            zs, widget._plot._display_points._df["tvd"].to_list()
        )

        # Points will be displayed in IJK coordinates when the toggle is unchecked:
        widget._filter_panel._toggle_utm_coords.setChecked(False)
        assert_frame_equal(
            widget._plot._display_points._df,
            expected_displayed_points_ijk(),
        )

        # Unchecking the load RFT file toggle will clear the file responses:
        widget._load_rft_file_toggle.setChecked(False)
        assert widget._file_responses.is_empty()


def _mock_rft_ensemble(stored_observations, stored_location_metadata, stored_responses):
    ensemble = MagicMock()
    experiment = MagicMock()
    experiment.observations = (
        {} if stored_observations is None else {"rft": stored_observations}
    )
    type(ensemble).experiment = PropertyMock(return_value=experiment)
    if isinstance(stored_location_metadata, BaseException):
        ensemble.load_observation_location_metadata.side_effect = (
            stored_location_metadata
        )
    else:
        ensemble.load_observation_location_metadata.return_value = (
            pl.DataFrame(schema=RFTConfig.location_metadata_schema())
            if stored_location_metadata is None
            else stored_location_metadata
        )

    ensemble.add_rft_metadata_and_qc = partial(
        LocalEnsemble.add_rft_metadata_and_qc, ensemble
    )
    if isinstance(stored_responses, BaseException):
        ensemble.load_responses.side_effect = stored_responses
    else:
        ensemble.load_responses.return_value = (
            pl.DataFrame(schema=RFTConfig.response_schema())
            if stored_responses is None
            else stored_responses
        )
    return ensemble


@pytest.mark.parametrize(
    "eclbase",
    [
        "file-<IENS>-<ITER>.DATA",
        "file-<IENS>-<ITER>",
    ],
)
@pytest.mark.filterwarnings("ignore:.*contains a RFT key but no forward model step")
def test_that_response_file_path_is_resolved(qtbot, use_tmpdir, eclbase):
    ert_config = ErtConfig.from_dict(
        {
            "RUNPATH": "runpath/realization-<IENS>/iter-<ITER>",
            "ECLBASE": eclbase,
            "RFT": [
                {
                    "WELL": "*",
                    "DATE": "*",
                    "PROPERTIES": "*",
                }
            ],
        }
    )
    ensemble = MagicMock()
    ensemble.iteration = 0
    ensemble.experiment.response_configuration = (
        ert_config.ensemble_config.response_configs
    )
    widget = RftQcWidget(ert_config=ert_config)

    widget.update_realization(ensemble, 1)

    assert widget._current_realization is not None
    assert widget._current_realization.rft_file_path == (
        Path("runpath/realization-1/iter-0/file-1-0.RFT").resolve()
    )


def test_that_missing_response_file_shows_warning(qtbot):
    ensemble = _mock_rft_ensemble(
        stored_observations=None,
        stored_location_metadata=None,
        stored_responses=KeyError("rft"),
    )

    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget.update_realization(ensemble, 0)

    assert widget._responses.is_empty()
    assert widget._load_status_label.text().startswith("Could not load responses")


def test_that_malformed_response_file_shows_warning(qtbot):
    ensemble = _mock_rft_ensemble(
        stored_observations=None,
        stored_location_metadata=None,
        stored_responses=pl.DataFrame({"unexpected": [1.0]}),
    )

    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget.update_realization(ensemble, 0)

    assert widget._responses.is_empty()
    assert widget._load_status_label.text().startswith("Could not load responses")


def test_that_empty_response_loads_without_warning(qtbot):
    ensemble = _mock_rft_ensemble(
        stored_observations=None,
        stored_location_metadata=None,
        stored_responses=pl.DataFrame(schema=RFTConfig.response_schema()),
    )

    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget.update_realization(ensemble, 0)

    assert widget._responses.is_empty()
    assert not widget._load_status_label.text()


@pytest.mark.parametrize(
    "stored_observations",
    [
        pytest.param(
            None,
            id="no rft observations",
        ),
        pytest.param(
            pl.DataFrame(schema=_rft_observation_schema()),
            id="empty rft observations",
        ),
    ],
)
@pytest.mark.parametrize(
    "stored_location_metadata",
    [
        pytest.param(
            pl.DataFrame(schema=RFTConfig.location_metadata_schema()),
            id="empty observation metadata",
        ),
        pytest.param(
            FileNotFoundError(),
            id="no observation metadata file",
        ),
    ],
)
def test_that_no_observations_loads_without_warning(
    qtbot, stored_observations, stored_location_metadata
):
    ensemble = _mock_rft_ensemble(
        stored_observations=stored_observations,
        stored_location_metadata=stored_location_metadata,
        stored_responses=pl.DataFrame(schema=RFTConfig.response_schema()),
    )

    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget.update_realization(ensemble, 0)

    assert widget._observations.is_empty()
    assert not widget._load_status_label.text()


def test_that_malformed_observations_shows_warning(qtbot):
    ensemble = _mock_rft_ensemble(
        stored_observations=pl.DataFrame({"unexpected": [1.0]}),
        stored_location_metadata=pl.DataFrame(
            schema=RFTConfig.location_metadata_schema()
        ),
        stored_responses=pl.DataFrame(schema=RFTConfig.response_schema()),
    )

    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget.update_realization(ensemble, 0)

    assert widget._observations.is_empty()
    assert widget._load_status_label.text().startswith("Could not load observations")


def test_that_invalid_rft_file_reports_error_and_keeps_file_responses_empty(
    qtbot, tmp_path
):
    rft_file = tmp_path / "BASE.RFT"
    rft_file.write_bytes(b"")

    rft_config = MagicMock(spec=RFTConfig)
    rft_config.read_from_file.side_effect = InvalidResponseFile(
        "BASE.RFT could not be parsed as an RFT file"
    )

    widget = RftQcWidget()
    widget._get_runpath = lambda ensemble, realization: tmp_path
    widget._current_realization = CurrentRealization(
        ensemble=MagicMock(),
        number=0,
        runpath=tmp_path,
        rft_config=rft_config,
        loaded=True,
    )

    widget._on_toggle_file_rft(True)

    assert widget._file_responses.is_empty()
    assert widget._rft_file_label.text() == (
        "BASE.RFT could not be parsed as an RFT file"
    )


def test_that_missing_rft_file_keeps_load_toggle_disabled(qtbot):
    widget = RftQcWidget()
    widget._current_realization = CurrentRealization(
        ensemble=MagicMock(),
        number=0,
        runpath="/does/not/exist",
        rft_config=RFTConfig(
            input_files=["BASE"],
            data_to_read={"*": {"*": ["*"]}},
            zonemap=None,
        ),
        loaded=True,
    )

    widget._update_load_rft_file_toggle_enabled_state()

    assert not widget._load_rft_file_toggle.isEnabled()


def test_that_observations_status_column_is_added(qtbot):
    # fmt: off
    observations = [
        # east  north   tvd   zone    observation
        ( 1.0,   1.0,   1.0, "zone1",  10.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 1.0,   1.0,   2.0,    None,  11.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma, multiple-spaces-before-keyword]
        ( 1.0,   1.0,   3.0,    None,  12.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma, multiple-spaces-before-keyword]
        ( 2.0,   1.0,   1.0, "zone1",  13.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 2.0,   1.0,   3.0, "zone1",  14.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 3.0,   1.0,   1.0, "zone1",  15.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 4.0,   1.0,   1.0, "zone1",  16.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ]

    metadata = [
        # east  north   tvd   actual_zones  connection_cell   cell_center
        ( 1.0,   1.0,   1.0, ["zone1"],       [1, 1, 1],    [1.0, 1.0, 1.0]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 1.0,   1.0,   2.0,        [],       [1, 1, 2],    [1.0, 1.0, 2.0]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 1.0,   1.0,   3.0, ["zone1"],       [1, 1, 3],    [1.0, 1.0, 3.0]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 2.0,   1.0,   1.0,        [],       [2, 1, 1],    [2.0, 1.0, 1.0]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 2.0,   1.0,   3.0, ["zone2"],       [2, 1, 3],    [2.0, 1.0, 3.0]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 3.0,   1.0,   1.0,        [],            None,               None),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma, multiple-spaces-before-keyword]
        ( 4.0,   1.0,   1.0, ["zone1"],       [4, 1, 1],    [4.0, 1.0, 1.0]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ]

    responses = [
        # connection_cell  values   cell_center      depth  cell_zones
        ( [1, 1, 1],       10.0,   [1.0, 1.0, 1.0],   1.0,  ["zone1"]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( [1, 1, 2],       11.0,   [1.0, 1.0, 2.0],   2.0,         []),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( [1, 1, 3],       12.0,   [1.0, 1.0, 3.0],   3.0,  ["zone1"]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( [2, 1, 1],       13.0,   [2.0, 1.0, 1.0],   1.0,         []),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( [2, 1, 3],       14.0,   [2.0, 1.0, 3.0],   3.0,  ["zone2"]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ]
    # fmt: on

    stored_observations = pl.DataFrame(
        observations,
        schema={
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "zone": pl.String,
            "observations": pl.Float32,
        },
        orient="row",
    ).with_columns(
        pl.lit("WELL:2000-01-01:PRESSURE").alias("response_key"),
        pl.lit("WELL").alias("well"),
        pl.lit("2000-01-01").alias("date"),
        pl.lit(5.0, dtype=pl.Float32).alias("std"),
    )

    stored_location_metadata = pl.DataFrame(
        metadata,
        schema=RFTConfig.location_metadata_schema(),
        orient="row",
    )

    stored_responses = pl.DataFrame(
        responses,
        schema={
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "values": pl.Float32,
            "cell_center": pl.Array(pl.Float32, 3),
            "depth": pl.Float32,
            "cell_zones": pl.List(pl.String),
        },
        orient="row",
    ).with_columns(
        pl.lit("WELL:2000-01-01:PRESSURE").alias("response_key"),
        pl.lit("WELL").alias("well"),
        pl.lit("2000-01-01").alias("date"),
        pl.lit("PRESSURE").alias("property"),
    )

    ensemble = _mock_rft_ensemble(
        stored_observations=stored_observations,
        stored_location_metadata=stored_location_metadata,
        stored_responses=stored_responses,
    )

    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget.update_realization(ensemble, 0)

    assert widget._observations["status"].to_list() == [
        _PointStatus.MATCHED,
        _PointStatus.MATCHED,
        _PointStatus.MATCHED,
        _PointStatus.INVALID_ZONE,
        _PointStatus.INVALID_ZONE,
        _PointStatus.NOT_IN_GRID,
        _PointStatus.NO_RESPONSE,
    ]


def test_that_deduplicate_points_per_coordinate_and_status_keeps_one_point_per_coordinate_per_status():  # ruff: ignore[line-too-long]
    # Every distinct status at a coordinate is kept, while duplicate rows sharing
    # both coordinate and status collapse to a single point.
    rows = (
        [(1.0, 1.0, 1.0, status) for status in _PointStatus] * 2
        + [(2.0, 2.0, 2.0, _PointStatus.RESPONSE)] * 3
        + [(2.0, 2.0, 2.0, _PointStatus.FILE_RFT)]
    )
    random.shuffle(rows)  # The order of the input rows must not affect the result.

    df = pl.DataFrame(
        rows,
        schema={
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "status": pl.String,
        },
        orient="row",
    )

    expected_points = pl.DataFrame(
        [(1.0, 1.0, 1.0, status.value) for status in _PointStatus]
        + [
            (2.0, 2.0, 2.0, _PointStatus.RESPONSE.value),
            (2.0, 2.0, 2.0, _PointStatus.FILE_RFT.value),
        ],
        schema={
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "status": pl.String,
        },
        orient="row",
    )

    assert_frame_equal(
        _deduplicate_points_per_coordinate_and_status(df),
        expected_points,
        check_row_order=False,
    )


def _two_point_ensemble(*, with_observations: bool = True, with_responses: bool = True):
    # fmt: off
    observations = [
        # east  north   tvd   zone    observation
        ( 1.0,   1.0,   1.0, "zone1",  10.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 1.0,   1.0,   2.0, "zone2",  13.0),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ] if with_observations else []
    metadata = [
        # east  north   tvd   actual_zones  connection_cell   cell_center
        ( 1.0,   1.0,   1.0, ["zone1"],       [1, 1, 1],    [1.0, 1.0, 1.0]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( 1.0,   1.0,   2.0, ["zone2"],       [1, 1, 2],    [1.0, 1.0, 2.0]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ] if with_observations else []
    responses = [
        # connection_cell  values  depth  cell_center  cell_zones
        ( [1, 1, 1],       11.0,   1.1,  [1.0, 1.0, 1.0], ["zone1"]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( [1, 1, 2],       14.0,   2.1,  [1.0, 1.0, 2.0], ["zone2"]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ] if with_responses else []
    # fmt: on
    stored_observations = pl.DataFrame(
        observations,
        schema={
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "zone": pl.String,
            "observations": pl.Float32,
        },
        orient="row",
    ).with_columns(
        pl.lit("WELL:2000-01-01:PRESSURE").alias("response_key"),
        pl.lit("WELL").alias("well"),
        pl.lit("2000-01-01").alias("date"),
        pl.lit(5.0, dtype=pl.Float32).alias("std"),
    )
    stored_location_metadata = pl.DataFrame(
        metadata,
        schema=RFTConfig.location_metadata_schema(),
        orient="row",
    )
    stored_responses = pl.DataFrame(
        responses,
        schema={
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "values": pl.Float32,
            "depth": pl.Float32,
            "cell_center": pl.Array(pl.Float32, 3),
            "cell_zones": pl.List(pl.String),
        },
        orient="row",
    ).with_columns(
        pl.lit("WELL:2000-01-01:PRESSURE").alias("response_key"),
        pl.lit("WELL").alias("well"),
        pl.lit("2000-01-01").alias("date"),
        pl.lit("PRESSURE").alias("property"),
    )
    return _mock_rft_ensemble(
        stored_observations=stored_observations,
        stored_location_metadata=stored_location_metadata,
        stored_responses=stored_responses,
    )


def _two_point_file_response():
    # fmt: off
    # Note: The file responses are different from the stored responses to ensure that
    # the widget is actually reading from the file and not just using the stored
    # response.
    responses_from_file = [
        # connection_cell  values  depth  cell_center  cell_zones
        ( [1, 1, 1],       12.0,   1.1,  [1.0, 1.0, 1.0], ["zone1"]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
        ( [1, 1, 2],       15.0,   2.1,  [1.0, 1.0, 2.0], ["zone2"]),  # ruff: ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    ]
    # fmt: on

    return pl.DataFrame(
        responses_from_file,
        schema={
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "values": pl.Float32,
            "depth": pl.Float32,
            "cell_center": pl.Array(pl.Float32, 3),
            "cell_zones": pl.List(pl.String),
        },
        orient="row",
    ).with_columns(
        pl.lit("WELL").alias("well"),
        pl.lit("2000-01-01").alias("date"),
        pl.lit("PRESSURE").alias("property"),
    )


def _rft_qc_widget(qtbot, ensemble, file_response=None):
    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget.update_realization(ensemble, 0)
    if file_response is not None:
        assert widget._current_realization is not None
        rft_config = MagicMock()
        rft_config.read_from_file.return_value = file_response
        rft_file_path = MagicMock()
        rft_file_path.exists.return_value = True
        widget._current_realization.runpath = "dummy/path"
        widget._current_realization.rft_config = rft_config
        widget._current_realization.rft_file_path = rft_file_path
    return widget


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _dispatch_mouse_event(
    plot: RftPlot, event_name: str, x: float, y: float, **event_kwargs: Any
) -> None:
    event = MouseEvent(event_name, plot._canvas, x, y, **event_kwargs)
    plot._canvas.callbacks.process(event_name, event)


def _point_display_xy(point_artist: Any, index: int):
    display_xy = point_artist.get_offset_transform().transform(
        point_artist.get_offsets()
    )
    return display_xy[index]


def pick_point(plot: RftPlot, index: int) -> None:
    # Dispatch a real left-click over the point so it travels through matplotlib's
    # picking machinery
    x, y = _point_display_xy(plot._point_artist, index)
    _dispatch_mouse_event(plot, "button_press_event", x, y, button=MouseButton.LEFT)


def test_that_observations_can_be_loaded_without_responses(qtbot):
    widget = _rft_qc_widget(
        qtbot, _two_point_ensemble(with_observations=True, with_responses=False)
    )
    assert widget._plot._display_points._df["status"].to_list() == [
        _PointStatus.NO_RESPONSE,
        _PointStatus.NO_RESPONSE,
    ]


def test_that_selecting_point_shows_point_details(qtbot):
    widget = _rft_qc_widget(
        qtbot, _two_point_ensemble(), file_response=_two_point_file_response()
    )

    plot = widget._plot
    pick_point(plot, 0)  # Pick first point

    point_0_details_only_storage = """
        Grid cell: i=1, j=1, k=1
        Zone: zone1
        Observations:
        WELL - 2000-01-01 - PRESSURE
        Observation: 10.00
        Response: 11.00
        Utm Coordinates: 1.00,1.00,1.00
        Error: 5.00
        Expected Zone: zone1
        Status: Observation with response
        Responses:
        Depth: 1.10
        WELL - 2000-01-01 - PRESSURE
        Response: 11.00
        Status: Response known by ERT
    """

    assert _normalize_ws(widget._details.toPlainText()) == _normalize_ws(
        point_0_details_only_storage
    )

    widget._on_toggle_file_rft(True)  # Load responses from file
    pick_point(plot, 1)  # Pick second point

    point_1_details_storage_and_file = """
        Grid cell: i=1, j=1, k=2
        Zone: zone2
        Observations:
        WELL - 2000-01-01 - PRESSURE
        Observation: 13.00
        Response: 14.00
        Utm Coordinates: 1.00,1.00,2.00
        Error: 5.00
        Expected Zone: zone2
        Status: Observation with response
        Responses:
        Depth: 2.10
        WELL - 2000-01-01 - PRESSURE
        Response: 14.00
        Status: Response known by ERT
        RFT in file:
        Depth: 2.10
        WELL - 2000-01-01 - PRESSURE
        File Response: 15.00
    """

    assert _normalize_ws(widget._details.toPlainText()) == _normalize_ws(
        point_1_details_storage_and_file
    )


def test_that_facet_decorations_show_counts_and_grey_out_empty_facets(qtbot):
    panel = PlotControlPanel(
        lambda *_: None, lambda: None, lambda: None, lambda: None, lambda _: None, False
    )
    qtbot.addWidget(panel)
    df = pl.DataFrame({"well": ["A", "B"], "property": ["PRESSURE", "SWAT"]})
    panel.populate_filters([df])

    for i in range(panel._property_list.count()):
        item = panel._property_list.item(i)
        item.setSelected(item.data(Qt.ItemDataRole.UserRole) == "PRESSURE")
    panel.refresh_facet_decorations()

    well_items = {}
    for i in range(panel._well_list.count()):
        item = panel._well_list.item(i)
        well_items[item.data(Qt.ItemDataRole.UserRole)] = item

    assert well_items["A"].text() == "A  (1)"
    assert well_items["B"].text() == "B  (0)"

    # The facet with no matching rows is greyed out, the populated one is not.
    grey = QColor(GREY)
    assert well_items["A"].foreground().color() != grey
    assert well_items["B"].foreground().color() == grey


def test_that_utm_toggle_is_unchecked_when_coordinates_become_unavailable(qtbot):
    panel = PlotControlPanel(
        lambda *_: None, lambda: None, lambda: None, lambda: None, lambda _: None, True
    )
    qtbot.addWidget(panel)
    assert panel._toggle_utm_coords.isChecked()

    panel.update_utm_available(False)

    assert not panel._toggle_utm_coords.isChecked()
    assert not panel._toggle_utm_coords.isEnabled()


def test_that_runpath_is_resolved_from_ert_config(qtbot):
    ert_config = ErtConfig.from_dict({"RUNPATH": "runpath/real-<IENS>/iter-<ITER>"})
    widget = RftQcWidget(ert_config)

    ensemble = MagicMock()
    ensemble.iteration = 0
    widget.update_realization(ensemble, 1)

    runpath = widget._get_runpath(ensemble, 1)

    assert runpath is not None
    assert runpath.endswith("runpath/real-1/iter-0")


def test_that_utm_is_reset_and_made_unavailable_when_cell_center_info_is_missing(qtbot):
    # Legacy ert runs did not store cell_center information for rft responses
    # The RFT QC tool handles this by disabling the UTM toggle when cell_center
    # information is missing.
    widget = RftQcWidget()
    widget._use_utm = True

    # Loading an ensemble with a response that has no cell_center information:
    ensemble = _mock_rft_ensemble(
        stored_observations=None,
        stored_location_metadata=None,
        stored_responses=pl.DataFrame(
            {
                "response_key": ["WELL:2000-01-01:PRESSURE"],
                "well": ["WELL"],
                "date": ["2000-01-01"],
                "property": ["PRESSURE"],
                "time": [None],
                "depth": pl.Series([50.0], dtype=pl.Float32),
                "values": pl.Series([100.0], dtype=pl.Float32),
                "well_connection_cell": pl.Series(
                    [[1, 1, 1]], dtype=pl.Array(pl.Int64, 3)
                ),
                "cell_zones": [["zone1"]],
            }
        ),
    )
    widget.update_realization(ensemble, 1)

    # The UTM toggle is reset and the option to enable it is made unavailable
    widget._use_utm = False
    assert not widget._filter_panel._toggle_utm_coords.isChecked()
    assert not widget._filter_panel._toggle_utm_coords.isEnabled()


def test_that_reapplying_filter_preserves_view_and_fit_restores_it(qtbot):
    widget = _rft_qc_widget(qtbot, _two_point_ensemble())

    # Simulating the user interacting with the plot limits through zooming and panning
    ax = widget._plot._ax
    original_ax_limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1, 1)

    widget._apply_filter_and_redraw()

    assert ax.get_xlim() == pytest.approx((-5, 5))
    assert ax.get_ylim() == pytest.approx((-3, 3))
    assert ax.get_zlim() == pytest.approx((-1, 1))

    # Simulating the user clicking the "Fit" button to
    # restore the view to autoscaled limits based on the filtered data
    widget._filter_panel._fit_button.click()

    np.testing.assert_allclose(
        (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()), original_ax_limits
    )


def test_that_empty_redraw_clears_points_and_preserves_view_limits(qtbot):
    plot = RftPlot(lambda *_: None, lambda: None)
    empty = pl.DataFrame(schema=DisplayPoints.schema())
    points = _display_points_df(
        [
            ([1, 1, 1], 1.0, 1.0, 1.0, _PointStatus.MATCHED),
            ([1, 1, 2], 1.0, 1.0, 2.0, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)

    # Simulating the user interacting with the plot limits through zooming and panning
    plot._ax.set_xlim(-5, 5)
    plot._ax.set_ylim(-3, 3)
    plot._ax.set_zlim(-1, 1)

    # Redrawing with empty data clears the points and preserves limits
    plot.redraw(empty, empty, empty)

    assert plot._point_artist is None
    assert plot._ax.get_xlim() == pytest.approx((-5, 5))
    assert plot._ax.get_ylim() == pytest.approx((-3, 3))
    assert plot._ax.get_zlim() == pytest.approx((-1, 1))


def test_that_center_on_selected_recenters_view_on_selected_point(qtbot):
    widget = _rft_qc_widget(qtbot, _two_point_ensemble())
    plot = widget._plot
    ax = plot._ax
    original_ax_limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())

    # no point selected, so view is unchanged
    plot._center_on_selected()
    np.testing.assert_allclose(
        (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()), original_ax_limits
    )

    pick_point(plot, 0)  # Pick first point
    assert plot._selected_index == 0

    plot._center_on_selected()

    # The view limits should be centered on the first point's coordinates
    coordinates = plot._display_points.get_point_coordinates(0)
    assert coordinates is not None
    cx, cy, cz = coordinates
    for (lo, hi), c in zip(
        (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()), (cx, cy, cz), strict=True
    ):
        assert (lo + hi) / 2 == pytest.approx(c)

    pick_point(plot, 1)  # Pick second point
    assert plot._selected_index == 1

    plot._center_on_selected()

    # The view limits should be centered on the second point's coordinates
    coordinates = plot._display_points.get_point_coordinates(1)
    assert coordinates is not None
    cx, cy, cz = coordinates
    for (lo, hi), c in zip(
        (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()), (cx, cy, cz), strict=True
    ):
        assert (lo + hi) / 2 == pytest.approx(c)


def test_that_clicking_away_from_points_does_not_select_anything(qtbot):
    widget = _rft_qc_widget(qtbot, _two_point_ensemble())
    plot = widget._plot

    # Clicking inside the axes but away from any point does not pick anything.
    corner_x, corner_y = plot._ax.transAxes.transform((0.02, 0.02))
    _dispatch_mouse_event(
        plot, "button_press_event", corner_x, corner_y, button=MouseButton.LEFT
    )
    assert plot._selected_index is None

    # Clicking outside the axes does not pick anything either.
    _dispatch_mouse_event(
        plot, "button_press_event", -100.0, -100.0, button=MouseButton.LEFT
    )
    assert plot._selected_index is None


def _display_points_df(
    rows: list[tuple[list[int], float | None, float | None, float | None, str]],
) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=DisplayPoints.schema(), orient="row")


def test_that_selected_point_stays_selected_after_redraw_when_point_is_unchanged(qtbot):
    plot = RftPlot(lambda *_: None, lambda: None)
    empty = pl.DataFrame(schema=DisplayPoints.schema())
    points = _display_points_df(
        [
            ([1, 1, 1], 1.0, 1.0, 1.0, _PointStatus.MATCHED),
            ([1, 1, 2], 1.0, 1.0, 2.0, _PointStatus.MATCHED),
            ([1, 1, 2], 9.0, 9.0, 9.0, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)
    index = 1
    pick_point(plot, index)
    assert plot._selected_index == index

    # Redraw same points in different order
    points = _display_points_df(
        [
            ([1, 1, 2], 1.0, 1.0, 2.0, _PointStatus.MATCHED),
            ([1, 1, 1], 1.0, 1.0, 1.0, _PointStatus.MATCHED),
            ([1, 1, 2], 9.0, 9.0, 9.0, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)

    # Point selection is the same, but index has changed
    assert plot._selected_index == 0


def test_that_selection_falls_back_to_same_cell_when_coordinates_change_after_redraw(
    qtbot,
):
    plot = RftPlot(lambda *_: None, lambda: None)
    empty = pl.DataFrame(schema=DisplayPoints.schema())
    points = _display_points_df(
        [
            ([1, 1, 1], 1.0, 1.0, 1.0, _PointStatus.MATCHED),
            ([1, 1, 2], 1.0, 1.0, 2.0, _PointStatus.MATCHED),
            ([1, 1, 2], 9.0, 9.0, 9.0, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)
    index = 1
    pick_point(plot, index)
    assert plot._selected_index == index

    # The point keeps its cell but is drawn at new coordinates after the redraw.
    points = _display_points_df(
        [
            ([1, 1, 2], 9.0, 9.0, 9.0, _PointStatus.MATCHED),
            ([1, 1, 1], 1.0, 1.0, 1.0, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)

    # Exact match no longer exist, so selection falls back to the same cell
    assert plot._selected_index == 0


def test_that_selection_is_cleared_when_selected_cell_is_absent_after_redraw(qtbot):
    plot = RftPlot(lambda *_: None, lambda: None)
    empty = pl.DataFrame(schema=DisplayPoints.schema())
    points = _display_points_df(
        [
            ([1, 1, 1], 1.0, 1.0, 1.0, _PointStatus.MATCHED),
            ([1, 1, 2], 1.0, 1.0, 2.0, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)
    index = 1
    pick_point(plot, index)
    assert plot._selected_index == index

    points = _display_points_df(
        [
            ([1, 1, 1], 1.0, 1.0, 1.0, _PointStatus.MATCHED),
            ([1, 1, 3], 9.0, 9.0, 9.0, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)

    # The selected cell is no longer present, so selection is cleared
    assert plot._selected_index is None


def test_that_selected_point_is_kept_based_on_cell_index_if_utm_coords_missing(qtbot):
    plot = RftPlot(lambda *_: None, lambda: None)
    empty = pl.DataFrame(schema=DisplayPoints.schema())
    points = _display_points_df(
        [
            ([1, 1, 1], None, None, None, _PointStatus.MATCHED),
            ([1, 1, 2], None, None, None, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)
    index = 1
    pick_point(plot, index)
    assert plot._selected_index == index

    # Redraw same points in different order
    points = _display_points_df(
        [
            ([1, 1, 2], None, None, None, _PointStatus.MATCHED),
            ([1, 1, 1], None, None, None, _PointStatus.MATCHED),
        ]
    )
    plot.redraw(points, empty, empty)

    # Point selection is the same, but index has changed
    assert plot._selected_index == 0


def test_that_scrolling_over_a_point_does_not_select_it(qtbot):
    widget = _rft_qc_widget(qtbot, _two_point_ensemble())
    plot = widget._plot

    x, y = _point_display_xy(plot._point_artist, 0)

    # Matplotlib routes scroll events through the same picking machinery as clicks,
    # so a real scroll over a point must not select it.
    scroll_plot(plot, x, y, "up")
    assert plot._selected_index is None

    # A real click at the same location does select the point, proving the scroll
    # event above actually reached a pickable point.
    _dispatch_mouse_event(plot, "button_press_event", x, y, button=MouseButton.LEFT)
    assert plot._selected_index == 0


def hover_plot(plot: RftPlot, x: float, y: float) -> None:
    _dispatch_mouse_event(plot, "motion_notify_event", x, y)


def test_that_hovering_over_point_highlights_it_and_moving_away_clears_it(qtbot):
    widget = _rft_qc_widget(qtbot, _two_point_ensemble())
    plot = widget._plot
    artist = plot._point_artist
    display_xy = artist.get_offset_transform().transform(artist.get_offsets())

    # There are 4 points in the plot 2 observations and 2 responses
    assert display_xy.shape[0] == 4

    (xs, ys, zs) = plot._display_points.get_points_to_plot()
    for (i, (x, y)), cx, cy, cz in zip(enumerate(display_xy), xs, ys, zs, strict=True):
        hover_plot(plot, x, y)  # Hover over the point

        # Hovered point is overlaid:
        assert display_xy[plot._hover_index].tolist() == display_xy[i].tolist()
        assert plot._hover_artist._offsets3d == pytest.approx(([cx], [cy], [cz]))

        hover_plot(plot, x + 1000, y + 1000)  # Move away from the point

        # No hover overlay:
        assert plot._hover_index is None
        assert plot._hover_artist._offsets3d == ([], [], [])

    hover_plot(plot, -100.0, -100.0)  # Hover outside the axes

    # No hover overlay:
    assert plot._hover_index is None
    assert plot._hover_artist._offsets3d == ([], [], [])


def scroll_plot(
    plot: RftPlot, x: float, y: float, direction: Literal["up", "down"] = "up"
) -> None:
    # Dispatch a real scroll event; matplotlib derives the wheel direction from step.
    _dispatch_mouse_event(
        plot, "scroll_event", x, y, step=1 if direction == "up" else -1
    )


def test_that_scrolling_zooms_the_view(qtbot):
    widget = _rft_qc_widget(qtbot, _two_point_ensemble())
    plot = widget._plot

    def get_limits():
        return (*plot._ax.get_xlim(), *plot._ax.get_ylim(), *plot._ax.get_zlim())

    inside_x, inside_y = _point_display_xy(plot._point_artist, 0)

    # Scroll outside axes, should be ignored
    before_zoom = get_limits()
    scroll_plot(plot, -100.0, -100.0, "up")
    assert get_limits() == before_zoom

    # Scroll up, should zoom in
    before_zoom = get_limits()
    scroll_plot(plot, inside_x, inside_y, "up")
    after_zoom = get_limits()
    assert after_zoom != pytest.approx(before_zoom)

    # Scroll down, should zoom out
    scroll_plot(plot, inside_x, inside_y, "down")
    assert get_limits() == pytest.approx(before_zoom)

    # Scroll while a point is selected, should zoom in/out around that point
    pick_point(plot, 0)  # Pick first point
    scroll_plot(plot, inside_x, inside_y, "up")
    assert get_limits() != pytest.approx(before_zoom)
    # Since point is selected, limits after zoom is different than they were after
    # zooming without a point selected
    assert get_limits() != pytest.approx(after_zoom)


def test_that_point_details_omit_missing_fields_and_coordinates(qtbot):
    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget._observations = pl.DataFrame(schema=RftQcWidget._required_obs_subschema())
    widget._responses = pl.DataFrame(
        {
            "response_key": ["WELL:2000-01-01:PRESSURE"],
            "well": ["WELL"],
            "date": ["2000-01-01"],
            "property": ["PRESSURE"],
            "values": [100.0],
            "well_connection_cell": [[1, 1, 1]],
            "well_connection_cell_center": [[0.5, 0.5, 0.5]],
            "east": [100.0],
            "north": [200.0],
            "tvd": [300.0],
            "depth": [300.0],
            "cell_zones": [["zone1"]],
            "status": [_PointStatus.RESPONSE],
        },
        schema=RftQcWidget._required_response_subschema(),
    )

    widget._show_details(
        {
            "well_connection_cell": [1, 1, 1],
            "east": None,
            "north": None,
            "tvd": None,
        }
    )

    text = _normalize_ws(widget._details.toPlainText())
    assert "WELL - 2000-01-01 - PRESSURE" in text  # Assure details is not empty

    # Missing fields are omitted from the details text:
    assert "Observation:" not in text
    assert "Observation Coordinates" not in text
    assert "Error:" not in text
    assert "Expected Zone:" not in text


@pytest.mark.mpl_image_compare(tolerance=10.0, style="default")
@pytest.mark.usefixtures("use_tmpdir")
def test_that_rft_qc_plot_in_grid_coordinates_is_unchanged(
    qtbot, mocked_files, mock_resfo_file
):
    with _loaded_multi_well_rft_qc_widget(
        qtbot, mocked_files, mock_resfo_file
    ) as widget:
        _display_all_points(widget)
        return widget._plot.canvas.figure


@pytest.mark.mpl_image_compare(tolerance=10.0, style="default")
@pytest.mark.usefixtures("use_tmpdir")
def test_that_rft_qc_plot_in_utm_coordinates_is_unchanged(
    qtbot, mocked_files, mock_resfo_file
):
    with _loaded_multi_well_rft_qc_widget(
        qtbot, mocked_files, mock_resfo_file
    ) as widget:
        _display_all_points(widget)
        assert widget._filter_panel._toggle_utm_coords.isEnabled()
        widget._filter_panel._toggle_utm_coords.setChecked(True)
        return widget._plot.canvas.figure
