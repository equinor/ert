import shutil
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest
from resfo_utilities.testing import (
    Date,
    Simulator,
    Smspec,
    SmspecIntehead,
    SummaryMiniStep,
    SummaryStep,
    UnitSystem,
    Unsmry,
)

from ert.__main__ import run_convert_observations
from ert.config.observation_config_migrations import (
    remove_refcase_and_time_map_dependence_from_obs_config,
)
from ert.namespace import Namespace


def create_summary_smspec_unsmry(
    summary_vectors: dict[str, list[float]],
    start_date: datetime.date,
    time_step_in_days: float = 30,
):
    summary_keys = list(summary_vectors.keys())
    num_time_steps = len(summary_vectors[summary_keys[0]])

    unsmry = Unsmry(
        steps=[
            SummaryStep(
                seqnum=0,
                ministeps=[
                    SummaryMiniStep(
                        mini_step=0,
                        params=[
                            24 * time_step_in_days * step,
                            *[summary_vectors[key][step] for key in summary_vectors],
                        ],
                    )
                ],
            )
            for step in range(num_time_steps)
        ]
    )
    smspec = Smspec(
        nx=4,
        ny=4,
        nz=10,
        restarted_from_step=0,
        num_keywords=1 + len(summary_keys),
        restart="        ",
        keywords=["TIME    ", *summary_keys],
        well_names=[":+:+:+:+", *([":+:+:+:+"] * len(summary_keys))],
        region_numbers=[-32676, *([0] * len(summary_keys))],
        units=["HOURS   ", *(["SM3"] * len(summary_keys))],
        start_date=Date.from_datetime(start_date),
        intehead=SmspecIntehead(
            unit=UnitSystem.METRIC,
            simulator=Simulator.ECLIPSE_100,
        ),
    )

    return smspec, unsmry


@pytest.mark.usefixtures("use_tmpdir")
def test_that_history_observations_are_converted_to_summary_observations(tmp_path):
    # Create a refcase (SMSPEC/UNSMRY files)
    smspec, unsmry = create_summary_smspec_unsmry(
        summary_vectors={
            "FOPRH": [1] * 10,
            "FOPTH": [2] * 10,
            "FWPTH": [3] * 10,
        },
        start_date=datetime(2020, 1, 1),
        time_step_in_days=30,
    )

    smspec.to_file(tmp_path / "REFCASE.SMSPEC")
    unsmry.to_file(tmp_path / "REFCASE.UNSMRY")

    # Create a time map file
    time_map_path = Path("time_map.txt")
    time_map_path.write_text(
        dedent(
            """\
            2020-01-01
            2020-01-11
            2020-01-21
            """
        ),
        encoding="utf-8",
    )

    # Create observation config with HISTORY_OBSERVATION
    obs_config_path = Path("observations.txt")
    obs_config_path.write_text(
        dedent(
            """\
            HISTORY_OBSERVATION FOPR {
                ERROR = 0.1;
                ERROR_MODE = RELMIN;
                ERROR_MIN = 5.0;
            };
            """
        ),
        encoding="utf-8",
    )

    # Create ERT config
    config_path = Path("config.ert")
    config_path.write_text(
        dedent(
            """\
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            REFCASE REFCASE
            TIME_MAP time_map.txt
            OBS_CONFIG observations.txt
            """
        ),
        encoding="utf-8",
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config(str(config_path))

    assert result is not None
    assert Path(result.obs_config_path) == Path("./observations.txt").absolute()
    assert Path(result.refcase_path) == Path("./REFCASE").absolute()
    assert len(result.history_changes) == 1

    history_change = result.history_changes[0]
    assert history_change.source_observation.name == "FOPR"
    assert len(history_change.summary_obs_declarations) == 10

    # Each summary observation should have DATE, VALUE, ERROR, and KEY
    for declaration in history_change.summary_obs_declarations:
        assert "SUMMARY_OBSERVATION" in declaration
        assert "DATE" in declaration
        assert "VALUE" in declaration
        assert "ERROR" in declaration
        assert "KEY" in declaration


@pytest.mark.usefixtures("use_tmpdir")
def test_that_general_observations_with_date_use_restart_from_time_map():
    """
    Test that GENERAL_OBSERVATION with DATE is converted to use RESTART
    instead of relying on TIME_MAP.
    """
    # Create time map
    time_map_path = Path("time_map.txt")
    time_map_path.write_text(
        dedent(
            """\
            2020-01-01
            2020-01-11
            2020-01-21
            """
        ),
        encoding="utf-8",
    )

    # Create observation data file
    obs_data_path = Path("obs_data.txt")
    obs_data_path.write_text("1.0 0.1\n", encoding="utf-8")

    # Create observation config with GENERAL_OBSERVATION using DATE
    obs_config_path = Path("observations.txt")
    obs_config_path.write_text(
        dedent(
            """\
            GENERAL_OBSERVATION GEN_OBS {
                DATA = GEN_DATA;
                INDEX_LIST = 0;
                DATE = 2020-01-11;
                OBS_FILE = obs_data.txt;
            };
            """,
        ),
        encoding="utf-8",
    )

    # Create ERT config
    config_path = Path("config.ert")
    config_path.write_text(
        dedent(
            """\
            NUM_REALIZATIONS 1
            TIME_MAP time_map.txt
            GEN_DATA GEN_DATA RESULT_FILE:gen%d.txt REPORT_STEPS:0,1,2
            OBS_CONFIG observations.txt
            """,
        ),
        encoding="utf-8",
    )

    # Running the full migration via the CLI should now raise during parsing
    # because `DATE` is no longer allowed in `GENERAL_OBSERVATION`.
    Path("config.ert").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_convert_observations(Namespace(config=str(config_path)))
    assert (
        Path("observations.txt").read_text(encoding="utf-8")
        == """GENERAL_OBSERVATION GEN_OBS {
   DATA       = GEN_DATA;
   INDEX_LIST = 0;
   RESTART    = 1;
   OBS_FILE   = obs_data.txt;
};
"""
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_summary_observations_with_restart_use_date_from_refcase():
    """
    Test that SUMMARY_OBSERVATION with RESTART is converted to use DATE
    instead of relying on REFCASE/TIME_MAP.
    """
    # Create a refcase
    smspec, unsmry = create_summary_smspec_unsmry(
        summary_vectors={"FOPR": [100, 110, 120]}, start_date=datetime(2020, 1, 1)
    )
    smspec.to_file(Path("REFCASE.SMSPEC"))
    unsmry.to_file(Path("REFCASE.UNSMRY"))

    # Create observation config with SUMMARY_OBSERVATION using RESTART
    obs_config_path = Path("observations.txt")
    obs_config_path.write_text(
        dedent(
            """\
            SUMMARY_OBSERVATION FOPR_OBS {
                VALUE = 110.0;
                ERROR = 5.0;
                RESTART = 1;
                KEY = FOPR;
            };
            SUMMARY_OBSERVATION FOPR_OBS1 {
                VALUE = 112.0;
                ERROR = 5.0;
                RESTART = 2;
                KEY = FOPR;
            };
            SUMMARY_OBSERVATION FOPR_OBS2 {
                VALUE = 113.0;
                ERROR = 5.0;
                RESTART = 3;
                KEY = FOPR;
            };
            """
        ),
        encoding="utf-8",
    )

    # Create ERT config
    config_path = Path("config.ert")
    config_path.write_text(
        dedent(
            """\
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            REFCASE REFCASE
            OBS_CONFIG observations.txt
            """
        ),
        encoding="utf-8",
    )

    # Running the full migration via the CLI should now raise during parsing
    # because SUMMARY_OBSERVATION with RESTART conversion is unaffected, but
    # we rely on the CLI path for consistency with other tests.
    Path("config.ert").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_convert_observations(Namespace(config=str(config_path)))
    assert (
        Path("observations.txt").read_text(encoding="utf-8")
        == """SUMMARY_OBSERVATION FOPR_OBS {
   VALUE    = 110.0;
   ERROR    = 5.0;
   DATE     = 2020-01-01;
   KEY      = FOPR;
};
SUMMARY_OBSERVATION FOPR_OBS1 {
   VALUE    = 112.0;
   ERROR    = 5.0;
   DATE     = 2020-01-31;
   KEY      = FOPR;
};
SUMMARY_OBSERVATION FOPR_OBS2 {
   VALUE    = 113.0;
   ERROR    = 5.0;
   DATE     = 2020-03-01;
   KEY      = FOPR;
};
"""
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_history_summary_and_general_obs_are_all_migrated_together(tmp_path):
    start_date = datetime(2024, 1, 1)
    smspec, unsmry = create_summary_smspec_unsmry(
        summary_vectors={
            "FOPRH": [1.0] * 5,
            "FOPTH": [2.0] * 5,
            "FWPTH": [3.0] * 5,
            "WOPRH": [4.0] * 5,
        },
        start_date=start_date,
        time_step_in_days=10,
    )

    smspec.to_file(Path("REFCASE.SMSPEC"))
    unsmry.to_file(Path("REFCASE.UNSMRY"))

    Path("time_map.txt").write_text(
        dedent(
            """\
            2024-01-01
            2024-01-11
            2024-01-21
            2024-01-31
            2024-02-10
            """
        ),
        encoding="utf-8",
    )

    Path("gen_obs_1.txt").write_text("1.0 0.1\n", encoding="utf-8")
    Path("gen_obs_2.txt").write_text("2.0 0.2\n", encoding="utf-8")
    Path("gen_obs_3.txt").write_text("3.0 0.3\n", encoding="utf-8")

    # PS: Note the inserted
    # } }; { etc behind comments to guard against some edge cases
    # wrt block extraction
    Path("observations.txt").write_text(
        dedent(
            """HISTORY_OBSERVATION FOPR {
    ERROR = 0.1; -- { dummy
};

HISTORY_OBSERVATION FOPT {
    ERROR = 0.2; -- {{{{{};
};

GENERAL_OBSERVATION GEN_OBS_1 {
    DATA = MY_GEN_DATA;
    INDEX_LIST = 0; -- };
    DATE = 2024-01-11;
    OBS_FILE = gen_obs_1.txt;
};

GENERAL_OBSERVATION GEN_OBS_2 {
    DATA = MY_GEN_DATA;
    INDEX_LIST = 0;
    DATE = 2024-01-31;
    OBS_FILE = gen_obs_2.txt;
};

GENERAL_OBSERVATION GEN_OBS_NO_INDEX_LIST {
    DATA = MY_GEN_DATA;
    DATE = 2024-01-31;
    OBS_FILE = gen_obs_3.txt;
};

SUMMARY_OBSERVATION SUM_OBS_1 {
    VALUE = 150.0;
    ERROR = 15.0;
    RESTART = 2;
    KEY = FWPTH;
};

SUMMARY_OBSERVATION SUM_OBS_2 {
    VALUE = 250.0;
    ERROR = 25.0;
    RESTART = 4;
    KEY = WOPRH;
};
            """
        ),
        encoding="utf-8",
    )

    config_path = Path("config.ert")
    config_path.write_text(
        dedent(
            """\
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            REFCASE REFCASE
            TIME_MAP time_map.txt
            OBS_CONFIG observations.txt
            GEN_DATA MY_GEN_DATA RESULT_FILE:gen%d.txt REPORT_STEPS:0,1,2,3,4
            """
        ),
        encoding="utf-8",
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config(str(config_path))

    assert result is not None
    assert Path(result.obs_config_path).name == "observations.txt"
    assert Path(result.refcase_path).name == "REFCASE"

    assert len(result.history_changes) == 2
    assert [change.source_observation.name for change in result.history_changes] == [
        "FOPR",
        "FOPT",
    ]
    for history_change in result.history_changes:
        assert len(history_change.summary_obs_declarations) == 5
        for declaration in history_change.summary_obs_declarations:
            assert "SUMMARY_OBSERVATION" in declaration
            assert "DATE" in declaration
            assert "VALUE" in declaration
            assert "ERROR" in declaration
            assert "KEY" in declaration

    assert [
        (c.source_observation.name, c.source_observation.date, c.restart)
        for c in result.general_obs_changes
    ] == [
        ("GEN_OBS_1", "2024-01-11", 2),
        ("GEN_OBS_2", "2024-01-31", 4),
        ("GEN_OBS_NO_INDEX_LIST", "2024-01-31", 4),
    ]

    assert [
        (c.source_observation.name, c.source_observation.restart, c.date)
        for c in result.summary_obs_changes
    ] == [
        ("SUM_OBS_1", 2, datetime(2024, 1, 11, 0, 0)),
        ("SUM_OBS_2", 4, datetime(2024, 1, 31, 0, 0)),
    ]

    shutil.copy("observations.txt", "observations_edited.txt")
    result.apply_to_file(path=Path("observations_edited.txt"))
    edited_contents = Path("observations_edited.txt").read_text(encoding="utf-8")
    assert (
        edited_contents
        == """\
SUMMARY_OBSERVATION FOPR {
   VALUE    = 1.0;
   ERROR    = 0.10000000149011612;
   DATE     = 2024-01-01;
   KEY      = FOPR;
};

SUMMARY_OBSERVATION FOPR {
   VALUE    = 1.0;
   ERROR    = 0.10000000149011612;
   DATE     = 2024-01-11;
   KEY      = FOPR;
};

SUMMARY_OBSERVATION FOPR {
   VALUE    = 1.0;
   ERROR    = 0.10000000149011612;
   DATE     = 2024-01-21;
   KEY      = FOPR;
};

SUMMARY_OBSERVATION FOPR {
   VALUE    = 1.0;
   ERROR    = 0.10000000149011612;
   DATE     = 2024-01-31;
   KEY      = FOPR;
};

SUMMARY_OBSERVATION FOPR {
   VALUE    = 1.0;
   ERROR    = 0.10000000149011612;
   DATE     = 2024-02-10;
   KEY      = FOPR;
};

SUMMARY_OBSERVATION FOPT {
   VALUE    = 2.0;
   ERROR    = 0.4000000059604645;
   DATE     = 2024-01-01;
   KEY      = FOPT;
};

SUMMARY_OBSERVATION FOPT {
   VALUE    = 2.0;
   ERROR    = 0.4000000059604645;
   DATE     = 2024-01-11;
   KEY      = FOPT;
};

SUMMARY_OBSERVATION FOPT {
   VALUE    = 2.0;
   ERROR    = 0.4000000059604645;
   DATE     = 2024-01-21;
   KEY      = FOPT;
};

SUMMARY_OBSERVATION FOPT {
   VALUE    = 2.0;
   ERROR    = 0.4000000059604645;
   DATE     = 2024-01-31;
   KEY      = FOPT;
};

SUMMARY_OBSERVATION FOPT {
   VALUE    = 2.0;
   ERROR    = 0.4000000059604645;
   DATE     = 2024-02-10;
   KEY      = FOPT;
};

GENERAL_OBSERVATION GEN_OBS_1 {
   DATA       = MY_GEN_DATA;
   INDEX_LIST = 0;
   RESTART    = 2;
   OBS_FILE   = gen_obs_1.txt;
};

GENERAL_OBSERVATION GEN_OBS_2 {
   DATA       = MY_GEN_DATA;
   INDEX_LIST = 0;
   RESTART    = 4;
   OBS_FILE   = gen_obs_2.txt;
};

GENERAL_OBSERVATION GEN_OBS_NO_INDEX_LIST {
   DATA       = MY_GEN_DATA;
   RESTART    = 4;
   OBS_FILE   = gen_obs_3.txt;
};

SUMMARY_OBSERVATION SUM_OBS_1 {
   VALUE    = 150.0;
   ERROR    = 15.0;
   DATE     = 2024-01-11;
   KEY      = FWPTH;
};

SUMMARY_OBSERVATION SUM_OBS_2 {
   VALUE    = 250.0;
   ERROR    = 25.0;
   DATE     = 2024-01-31;
   KEY      = WOPRH;
};
"""
    )
