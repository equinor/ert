import io
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest

from ert.cli.main import ErtCliError
from ert.config import ErtConfig, ShapeRegistry
from ert.config._observations import BreakthroughObservation, SummaryObservation
from ert.config._shapes import CircleShapeConfig
from ert.observation_converters.summary_to_bulk_config import (
    BulkConfigConverter,
    convert_summary_to_bulk_config,
)


@pytest.mark.usefixtures("snake_oil_case")
def test_that_happy_path_on_snake_oil_produces_csv_and_stdout(capsys):
    """This tests that the produced stdout and csv file from the
    gather_summary_observations command is what we expect.
    Finally, the test also creates a ErtConfig from the initial observation config
    and compares it to the new one when replacing the summary observations with the
    stdout and moving the csv file into the observations folder.
    """
    config_path = "snake_oil.ert"
    convert_summary_to_bulk_config(config_path)

    assert Path("summary_observations.csv").is_file()
    csv_content = Path("summary_observations.csv").read_text(encoding="utf-8")
    expected_csv_content = [
        "keyword, well, value, error, date",
        "WOPR, OP1, 0.1, 0.05, 2010-03-31",
        "WOPR, OP1, 0.7, 0.07, 2010-12-26",
        "WOPR, OP1, 0.5, 0.05, 2011-12-21",
        "WOPR, OP1, 0.3, 0.075, 2012-12-15",
        "WOPR, OP1, 0.2, 0.035, 2013-12-10",
        "WOPR, OP1, 0.015, 0.01, 2015-03-15",
    ]

    assert all(line in csv_content for line in expected_csv_content)

    expected_stdout = "SUMMARY {\n  VALUES = summary_observations.csv;\n"
    stdout = capsys.readouterr().out
    assert expected_stdout in stdout

    observation_path = "observations/observations.txt"
    old_obs_config = Path(observation_path).read_text(encoding="utf-8")
    old_obs_config_lines = old_obs_config.split("\n")
    # Assumes General Observation at bottom of file
    summary_stop_line = next(
        i
        for i, line in enumerate(old_obs_config_lines)
        if "GENERAL_OBSERVATION" in line
    )
    non_bulk_observations = old_obs_config_lines[summary_stop_line:]

    stdout_lines = stdout.split("\n")
    bulk_start_line = next(
        i for i, line in enumerate(stdout_lines) if "SUMMARY {" in line
    )
    extracted_bulk_lines = stdout_lines[bulk_start_line:]

    new_obs_content = "\n".join([*extracted_bulk_lines, *non_bulk_observations])

    old_ert_config = ErtConfig.from_file("snake_oil.ert")

    Path(observation_path).write_text(new_obs_content, encoding="utf-8")
    shutil.move("summary_observations.csv", "observations/summary_observations.csv")
    new_ert_config = ErtConfig.from_file("snake_oil.ert")

    assert len(new_ert_config.observation_declarations) == len(
        old_ert_config.observation_declarations
    )
    # Loop through observations and assert that they are the same except name.
    # This also checks that the ordering is the same.
    for i in range(len(new_ert_config.observation_declarations)):
        old_obs = old_ert_config.observation_declarations[i].__dict__
        new_obs = new_ert_config.observation_declarations[i].__dict__
        # Bulk summary config must utilize default naming for observations,
        # so the names will differ between the observation declarations.
        old_obs.pop("name")
        new_obs.pop("name")
        assert old_obs == new_obs


@pytest.fixture(name="patched_csv_writer")
def patched_csv_writing(monkeypatch):
    """Avoid writing to file.
    Fixture mock can be used to assert what has been written to file.
    """
    write_buffer = io.StringIO()

    @contextmanager
    def mock_open(*args, **kwargs):
        yield write_buffer

    monkeypatch.setattr(Path, "open", mock_open)
    return write_buffer


def _make_summary_obs(
    key: str = "WOPR",
    well: str | None = "OP1",
    date: str = "2010-01-27",
    shape_id: int | None = None,
) -> SummaryObservation:
    key += f":{well}" if well else ""
    return SummaryObservation(
        name="foo",
        key=f"{key}:{well}",
        value=0.5,
        error=0.02,
        date=date,
        shape_id=shape_id,
    )


def _make_breakthrough_obs(
    well: str,
    date: str = "2010-02-27",
    shape_id: int | None = None,
) -> BreakthroughObservation:
    return BreakthroughObservation(
        name=f"BREAKTHROUGH_WWCT_{well}",
        key=f"WWCT:{well}",
        date=datetime.fromisoformat(date),
        error=0.02,
        threshold=0.5,
        shape_id=shape_id,
    )


@pytest.mark.usefixtures("patched_csv_writer")
def test_that_convert_summary_observations_extracts_localization_information(capsys):
    shape_registry = ShapeRegistry()

    shape_id_with_radius = shape_registry.register(
        CircleShapeConfig(east=10, north=20, radius=2500)
    )
    obs_with_loc = _make_summary_obs(
        well="WELL_WITH_LOCALIZATION", shape_id=shape_id_with_radius
    )

    BulkConfigConverter(
        [obs_with_loc],
        shape_registry,
    ).print_bulk_config()

    expected_print_with_localization = dedent("""\
    SUMMARY {
      VALUES = summary_observations.csv;
      WELL WELL_WITH_LOCALIZATION {
        LOCALIZATION {
          EAST=10;
          NORTH=20;
          RADIUS=2500;
        };
      };
    };""")

    assert expected_print_with_localization in capsys.readouterr().out


@pytest.mark.usefixtures("patched_csv_writer")
def test_that_convert_summary_observations_produces_natsorted_csv_rows(
    monkeypatch, patched_csv_writer
):
    observations = [
        _make_summary_obs("OP30"),
        _make_summary_obs("OP4"),
        _make_summary_obs("OP10"),
        _make_summary_obs("OP2"),
    ]

    BulkConfigConverter(
        observations=observations,
    ).write_csv()

    ordered_wells = ["OP2", "OP4", "OP10", "OP30"]
    csv_content = patched_csv_writer.getvalue()
    obs_rows = csv_content.strip().split("\n")[1:]
    csv_well_ordering = [row.split(",")[0].strip() for row in obs_rows]
    assert csv_well_ordering == ordered_wells


@pytest.mark.usefixtures("patched_csv_writer")
def test_that_convert_summary_observations_chronologically_sorts_within_well(
    monkeypatch, patched_csv_writer
):
    observations = [
        _make_summary_obs(well="OP1", date="2010-01-01"),
        _make_summary_obs(well="OP2", date="2010-01-03"),
        _make_summary_obs(well="OP1", date="2010-01-03"),
        _make_summary_obs(well="OP2", date="2010-01-01"),
        _make_summary_obs(well="OP1", date="2010-01-02"),
        _make_summary_obs(well="OP2", date="2010-01-02"),
    ]

    BulkConfigConverter(
        observations=observations,
    ).write_csv()

    ordered_wells = ["OP1"] * 3 + ["OP2"] * 3
    csv_content = patched_csv_writer.getvalue()
    obs_rows = csv_content.strip().split("\n")[1:]
    csv_well_ordering = [row.split(",")[1].strip() for row in obs_rows]
    assert csv_well_ordering == ordered_wells

    ordered_dates = ["2010-01-01", "2010-01-02", "2010-01-03"] * 2
    csv_date_ordering = [row.split(",")[4].strip() for row in obs_rows]
    csv_date_ordering = [d.split("T")[0] for d in csv_date_ordering]
    assert ordered_dates == csv_date_ordering


@pytest.mark.usefixtures("patched_csv_writer")
def test_that_localization_can_be_gathered_from_breakthrough(capsys):
    shape_registry = ShapeRegistry()
    shape_id = shape_registry.register(
        CircleShapeConfig(east=10, north=20, radius=2500)
    )

    summary_obs = _make_summary_obs()
    brt_obs = _make_breakthrough_obs("OP1", shape_id=shape_id)

    BulkConfigConverter(
        [summary_obs, brt_obs],
        shape_registry,
    ).print_bulk_config()

    assert (
        "  WELL OP1 {\n"
        "    LOCALIZATION {\n"
        "      EAST=10;\n"
        "      NORTH=20;\n"
        "      RADIUS=2500;\n"
        "    };\n"
    ) in capsys.readouterr().out


@pytest.mark.usefixtures("patched_csv_writer")
def test_that_multiple_breakthrough_observations_for_the_same_well_raises_cli_error(
    monkeypatch,
):
    brt1 = _make_breakthrough_obs("OP1", date="2010-02-27")
    brt2 = _make_breakthrough_obs("OP1", date="2010-03-27")

    with pytest.raises(
        ErtCliError,
        match=r"Can only have one breakthrough observation per well.\n"
        r"Found 2 breakthroughs for well 'OP1'.",
    ):
        BulkConfigConverter([brt1, brt2])


@pytest.mark.usefixtures("patched_csv_writer")
def test_that_the_correct_number_of_observations_are_mentioned_in_helper_text(capsys):
    observations = [
        _make_summary_obs(well="OP1"),
        _make_summary_obs(well="OP1"),
        _make_summary_obs(well="OP2"),
        _make_breakthrough_obs(well="OP2"),
    ]
    BulkConfigConverter(observations).print_bulk_config()
    assert "4 observations can be replaced" in capsys.readouterr().out


@pytest.mark.usefixtures("patched_csv_writer")
def test_that_bpr_observation_populates_ijk_columns_while_others_are_left_empty(
    patched_csv_writer,
):
    well_obs = _make_summary_obs()
    bpr_obs = _make_summary_obs(
        key="BPR:1,2,3",
    )

    BulkConfigConverter([well_obs, bpr_obs]).write_csv()

    csv_content = patched_csv_writer.getvalue()
    lines = csv_content.strip().split("\n")

    header = lines[0]
    expected_headers = ["keyword", "well", "i", "j", "k"]
    assert all(h in header for h in expected_headers)

    bpr_line = lines[1]
    assert "BPR, , 1, 2, 3, 0.5, 0.02, 2010-01-27" in bpr_line

    wopr_line = lines[2]
    assert "WOPR, OP1, , , , 0.5, 0.02, 2010-01-27" in wopr_line
