import io
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest

from ert.cli.main import ErtCliError
from ert.gather_summary_observations import (
    convert_summary_observations,
    fetch_experiments,
    get_experiment_id,
)


@pytest.fixture(name="patched_csv_writer")
def patch_csv_writing(monkeypatch):
    write_buffer = io.StringIO()

    @contextmanager
    def mock_open(*args, **kwargs):
        yield write_buffer

    monkeypatch.setattr(Path, "open", mock_open)
    return write_buffer


@pytest.mark.usefixtures("use_tmpdir")
def test_that_convert_summary_observations_extracts_localization_information(capsys):
    summary_obs = {
        "WOPR:WELL_WITH_LOCALIZATION": [
            {
                "name": "WOPR_OP1_72",
                "errors": [0.02],
                "values": [0.5],
                "east": [None],
                "north": [None],
                "radius": [None],
                "x_axis": ["2010-01-27T00:00:00"],
            },
            {
                "name": "WOPR_OP1_9",
                "errors": [0.05],
                "values": [0.1],
                "east": [10],
                "north": [20],
                "radius": [2500],
                "x_axis": ["2010-03-31T00:00:00"],
            },
        ],
        "WSPC:WELL_WITHOUT_RADIUS": [
            {
                "name": "WSPC",
                "errors": [0.04],
                "values": [3],
                "east": [40],
                "north": [50],
                "radius": [None],
                "x_axis": ["2011-12-21T00:00:00"],
            },
        ],
        "WSIR:WELL_WITHOUT_LOCALIZATION": [
            {
                "name": "WSIR",
                "errors": [0.04],
                "values": [3],
                "x_axis": ["2011-12-21T00:00:00"],
            },
        ],
    }
    convert_summary_observations(
        summary_observations=summary_obs,
        breakthrough_observations={},
        csv_file_name="foo.csv",
    )
    expected_print_with_localization = dedent("""\
    SUMMARY {
      VALUES = foo.csv;
      WELL WELL_WITHOUT_RADIUS {
        LOCALIZATION {
          EAST=40;
          NORTH=50;
        };
      };
      WELL WELL_WITH_LOCALIZATION {
        LOCALIZATION {
          EAST=10;
          NORTH=20;
          RADIUS=2500;
        };
      };
    };""")

    assert expected_print_with_localization in capsys.readouterr().out


def make_summary_obs(number: int, extra_entries=None):
    if extra_entries is None:
        extra_entries = {}
    return {
        f"WOPR:OP{number}": [
            {
                "name": f"WOPR_OP{number}",
                "errors": [0.02],
                "values": [0.5],
                "x_axis": ["2010-01-27T00:00:00"],
                **extra_entries,
            },
        ]
    }


@pytest.mark.usefixtures("use_tmpdir")
def test_that_convert_summary_observations_produces_natsorted_csv_rows(
    monkeypatch, patched_csv_writer
):

    summary_obs = (
        make_summary_obs(30)
        | make_summary_obs(10)
        | make_summary_obs(4)
        | make_summary_obs(2)
    )

    convert_summary_observations(
        summary_observations=summary_obs,
        breakthrough_observations={},
        csv_file_name="foo.csv",
    )

    ordered_wells = ["OP2", "OP4", "OP10", "OP30"]
    csv_content = patched_csv_writer.getvalue()
    csv_well_ordering = [
        line.split(",")[1].strip() for line in csv_content.strip().split("\n")[1:]
    ]
    assert csv_well_ordering == ordered_wells


@pytest.mark.usefixtures("use_tmpdir")
async def test_that_no_experiments_in_storage_raises_ert_cli_error(monkeypatch):
    class MockClient:
        async def get(_):
            return_mock = MagicMock()
            return_mock.text = "{}"
            return return_mock

    with pytest.raises(ErtCliError, match=r"Could not fetch experiments from storage."):
        await fetch_experiments(MockClient)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_single_experiment_in_storage_is_automatically_selected_given_no_experiment_arg(  # noqa: E501
    capsys,
):
    exp_id = "foo"
    get_experiment_id([{"id": exp_id}], None)
    assert (
        f"Gathering observations for sole experiment in storage: '{exp_id}'"
        in capsys.readouterr().out
    )


def test_that_given_experiment_id_not_in_storage_raises_cli_error():
    exp_id_arg = "bar"
    with pytest.raises(
        ErtCliError, match=rf"An experiment with id '{exp_id_arg}' does not exist."
    ):
        get_experiment_id([{"id": "foo", "name": "run_model"}], exp_id_arg)


@pytest.mark.usefixtures("patched_csv_writer")
def test_that_localization_can_be_gathered_from_breakthrough(capsys):
    summary_obs = make_summary_obs(
        1, {"east": [None], "north": [None], "radius": [None]}
    )
    brt_obs = {
        "BREAKTHROUGH:WWCT:OP1": [
            {
                "name": "BREAKTHROUGH_WWCT_OP1",
                "errors": [0.02],
                "values": [0.5],
                "x_axis": ["2010-02-27T00:00:00"],
                "east": [10],
                "north": [20],
                "radius": [2500],
            }
        ]
    }
    convert_summary_observations(summary_obs, brt_obs, "foo")
    assert (
        "  WELL OP1 {\n"
        "    LOCALIZATION {\n"
        "      EAST=10;\n"
        "      NORTH=20;\n"
        "      RADIUS=2500;\n"
        "    };\n"
    ) in capsys.readouterr().out


def test_that_multiple_breakthrough_observations_for_the_same_well_raises_cli_error(
    monkeypatch,
):
    summary_obs = {}
    brt_obs = _make_brt_obs(
        1,
    )
    obs_name = next(name for name in brt_obs)
    brt_obs[obs_name] *= 2
    with pytest.raises(
        ErtCliError,
        match=r"Can only have one breakthrough observation per well.\n"
        r"Found 2 breakthroughs for well 'OP1'.",
    ):
        convert_summary_observations(summary_obs, brt_obs, "foo")
