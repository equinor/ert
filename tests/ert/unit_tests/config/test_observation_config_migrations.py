import shutil
from collections.abc import Iterable
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

from ert.config import ConfigValidationError
from ert.config.observation_config_migrations import (
    remove_refcase_and_time_map_dependence_from_obs_config,
)
from ert.config.parsing.observations_parser import ObservationConfigError
from ert.observation_converters.history_to_summary import convert_history_to_summary


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


def setup_refcase_config(
    obs_config_content: str,
    summary_vectors: dict[str, Iterable[float]],
    start_date: datetime = datetime(2020, 1, 1),  # ruff: ignore[call-datetime-without-tzinfo]
    time_step_in_days: float = 10,
    extra_config_lines: str = "",
) -> None:
    smspec, unsmry = create_summary_smspec_unsmry(
        summary_vectors=summary_vectors,
        start_date=start_date,
        time_step_in_days=time_step_in_days,
    )
    smspec.to_file(Path("REFCASE.SMSPEC"))
    unsmry.to_file(Path("REFCASE.UNSMRY"))

    Path("observations.txt").write_text(dedent(obs_config_content), encoding="utf-8")

    config_lines = ["NUM_REALIZATIONS 1", "ECLBASE ECLIPSE_CASE", "REFCASE REFCASE"]
    if extra_config_lines:
        config_lines.append(extra_config_lines)
    config_lines.append("OBS_CONFIG observations.txt")
    Path("config.ert").write_text("\n".join(config_lines) + "\n", encoding="utf-8")


def setup_timemap_config(
    obs_config_content: str,
    *,
    time_map_content: str = "2020-01-01\n2020-01-11\n",
    extra_config_lines: str = "",
) -> None:
    Path("time_map.txt").write_text(time_map_content, encoding="utf-8")
    Path("observations.txt").write_text(dedent(obs_config_content), encoding="utf-8")

    config_lines = ["NUM_REALIZATIONS 1", "TIME_MAP time_map.txt"]
    if extra_config_lines:
        config_lines.append(extra_config_lines)
    config_lines.append("OBS_CONFIG observations.txt")
    Path("config.ert").write_text("\n".join(config_lines) + "\n", encoding="utf-8")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_history_observations_are_converted_to_summary_observations():
    Path("time_map.txt").write_text(
        "2020-01-01\n2020-01-11\n2020-01-21\n", encoding="utf-8"
    )
    setup_refcase_config(
        """\
        HISTORY_OBSERVATION FOPR {
            ERROR = 0.1;
            ERROR_MODE = RELMIN;
            ERROR_MIN = 5.0;
        };
        """,
        summary_vectors={
            "FOPRH": [1] * 10,
            "FOPTH": [2] * 10,
            "FWPTH": [3] * 10,
        },
        time_step_in_days=30,
        extra_config_lines="TIME_MAP time_map.txt",
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

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
    Path("obs_data.txt").write_text("1.0 0.1\n", encoding="utf-8")

    setup_timemap_config(
        """\
        GENERAL_OBSERVATION GEN_OBS {
            DATA = GEN_DATA;
            INDEX_LIST = 0;
            DATE = 2020-01-11;
            OBS_FILE = obs_data.txt;
        };
        """,
        time_map_content="2020-01-01\n2020-01-11\n2020-01-21\n",
        extra_config_lines="GEN_DATA GEN_DATA RESULT_FILE:gen%d.txt REPORT_STEPS:0,1,2",
    )

    # Running the full migration via the CLI should now raise during parsing
    # because `DATE` is no longer allowed in `GENERAL_OBSERVATION`.
    convert_history_to_summary("config.ert")
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
    setup_refcase_config(
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
        """,
        summary_vectors={"FOPR": [100, 110, 120]},
        time_step_in_days=30,
    )

    # Running the full migration via the CLI should now raise during parsing
    # because SUMMARY_OBSERVATION with RESTART conversion is unaffected, but
    # we rely on the CLI path for consistency with other tests.
    convert_history_to_summary("config.ert")
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
def test_that_history_summary_and_general_obs_are_all_migrated_together():
    Path("time_map.txt").write_text(
        "2024-01-01\n2024-01-11\n2024-01-21\n2024-01-31\n2024-02-10\n",
        encoding="utf-8",
    )

    Path("gen_obs_1.txt").write_text("1.0 0.1\n", encoding="utf-8")
    Path("gen_obs_2.txt").write_text("2.0 0.2\n", encoding="utf-8")
    Path("gen_obs_3.txt").write_text("3.0 0.3\n", encoding="utf-8")

    # PS: Note the inserted
    # } }; { etc behind comments to guard against some edge cases
    # wrt block extraction
    setup_refcase_config(
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
            """,
        summary_vectors={
            "FOPRH": [1.0] * 5,
            "FOPTH": [2.0] * 5,
            "FWPTH": [3.0] * 5,
            "WOPRH": [4.0] * 5,
        },
        start_date=datetime(2024, 1, 1),  # ruff: ignore[call-datetime-without-tzinfo]
        extra_config_lines=(
            "TIME_MAP time_map.txt\n"
            "GEN_DATA MY_GEN_DATA RESULT_FILE:gen%d.txt REPORT_STEPS:0,1,2,3,4"
        ),
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

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
        ("SUM_OBS_1", 2, datetime(2024, 1, 11, 0, 0)),  # ruff: ignore[call-datetime-without-tzinfo]
        ("SUM_OBS_2", 4, datetime(2024, 1, 31, 0, 0)),  # ruff: ignore[call-datetime-without-tzinfo]
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


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("obs_config_content", "match"),
    [
        pytest.param(
            "HISTORY_OBSERVATION FOPR { };",
            "REFCASE is required for HISTORY_OBSERVATION",
            id="history observation without refcase",
        ),
        pytest.param(
            """\
            SUMMARY_OBSERVATION WOPR_OP1_9 {
                VALUE   = 0.1;
                ERROR   = 0.05;
                RESTART = 9;
                KEY     = WOPR:OP1;
            };
            """,
            "Missing REFCASE or TIME_MAP for observations: WOPR_OP1_9",
            id="summary observation restart without time map",
        ),
    ],
)
def test_that_observation_config_without_refcase_or_time_map_raises_error(
    obs_config_content, match
):
    Path("observations.txt").write_text(dedent(obs_config_content), encoding="utf-8")
    Path("config.ert").write_text(
        "NUM_REALIZATIONS 1\nECLBASE ECLIPSE_CASE\nOBS_CONFIG observations.txt\n",
        encoding="utf-8",
    )

    with pytest.raises(ObservationConfigError, match=match):
        remove_refcase_and_time_map_dependence_from_obs_config("config.ert")


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("obs_config_content", "match"),
    [
        pytest.param(
            "HISTORY_OBSERVATION FGPR { };",
            "Key 'FGPRH' is not present in refcase",
            id="history observation key missing from refcase",
        ),
        pytest.param(
            "HISTORY_OBSERVATION FOPR { ERROR_MODE = REL; };",
            "Observation uncertainty must be given a strictly positive value",
            id="history observation non positive uncertainty",
        ),
        pytest.param(
            "HISTORY_OBSERVATION FOPR { SEGMENT SEG { STOP = 3; }; };",
            'Missing item "START"',
            id="history observation segment missing start",
        ),
        pytest.param(
            "HISTORY_OBSERVATION FOPR { SEGMENT SEG { START = 0; }; };",
            'Missing item "STOP"',
            id="history observation segment missing stop",
        ),
        pytest.param(
            "GENERAL_OBSERVATION GEN_OBS { DATE = 2020-01-11; };",
            'Missing item "DATA"',
            id="general observation missing data key",
        ),
        pytest.param(
            """\
            GENERAL_OBSERVATION GEN_OBS {
                DATA = GEN_DATA;
                DATE = 2020-01-11;
                OBS_FILE = does_not_exist.txt;
            };
            """,
            r"did not.*resolve to a valid path",
            id="general observation missing obs file",
        ),
        pytest.param(
            """\
            GENERAL_OBSERVATION GEN_OBS {
                DATA  = GEN_DATA;
                DATE  = 2020-01-11;
                VALUE = 1.0;
            };
            """,
            "ERROR must also be given",
            id="general observation value without error",
        ),
        pytest.param(
            "SUMMARY_OBSERVATION FOPR_OBS { RESTART = 1; ERROR = 5.0; KEY = FOPR; };",
            'Missing item "VALUE"',
            id="summary_observation_missing_value",
        ),
        pytest.param(
            "SUMMARY_OBSERVATION FOPR_OBS "
            "{ RESTART = 1; VALUE = 110.0; ERROR = 5.0; };",
            'Missing item "KEY"',
            id="summary observation missing key",
        ),
        pytest.param(
            "SUMMARY_OBSERVATION FOPR_OBS { RESTART = 1; VALUE = 110.0; KEY = FOPR; };",
            'Missing item "ERROR"',
            id="summary observation missing error",
        ),
        pytest.param(
            "SUMMARY_OBSERVATION FOPR_OBS { DAYS = -1; };",
            None,
            id="summary observation restart negative",
        ),
        pytest.param(
            """\
            SUMMARY_OBSERVATION FOPR_OBS {
                VALUE   = 0.1;
                ERROR   = 0.05;
                RESTART = 99;
                KEY     = FOPR;
            };
            """,
            "beyond the last report step of the REFCASE/TIME_MAP",
            id="summary observation restart beyond last report step",
        ),
        pytest.param(
            """\
            SUMMARY_OBSERVATION FOPR_OBS {
                VALUE = 0.1;
                ERROR = 0.05;
                DAYS  = 100;
                KEY   = FOPR;
            };
            """,
            "Could not find",
            id="summary observation date not found in time map",
        ),
    ],
)
def test_that_invalid_history_observation_refcase_usage_raises_error(
    obs_config_content, match
):
    setup_refcase_config(obs_config_content, summary_vectors={"FOPRH": (100.0, 0.0)})

    with pytest.raises(ObservationConfigError, match=match):
        remove_refcase_and_time_map_dependence_from_obs_config("config.ert")


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "obs_config_content",
    [
        pytest.param(
            "HISTORY_OBSERVATION FOPR { UNKNOWN_KEY = 1; };",
            id="history observation",
        ),
        pytest.param(
            "HISTORY_OBSERVATION FOPR { SEGMENT SEG { UNKNOWN_KEY = 1; }; };",
            id="history observation segment",
        ),
        pytest.param(
            "SUMMARY_OBSERVATION FOPR_OBS { RESTART = 1; UNKNOWN_KEY = 1; };",
            id="summary observation",
        ),
        pytest.param(
            """\
            GENERAL_OBSERVATION GEN_OBS {
                DATA = GEN_DATA;
                DATE = 2020-01-11;
                UNKNOWN_KEY = 1;
            };
            """,
            id="general observation",
        ),
    ],
)
def test_that_unknown_observation_key_raises_error(obs_config_content):
    setup_timemap_config(obs_config_content)

    with pytest.raises(ObservationConfigError, match="Unknown key"):
        remove_refcase_and_time_map_dependence_from_obs_config("config.ert")


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("obs_config_content", "summary_vectors", "expected_errors"),
    [
        pytest.param(
            "HISTORY_OBSERVATION FOPR { ERROR = 10.0; ERROR_MODE = ABS; };",
            {"FOPRH": [100.0]},
            ["10.0"],
            id="error mode abs gives constant uncertainty",
        ),
        pytest.param(
            "HISTORY_OBSERVATION FGPR { ERROR = 0.1; ERROR_MODE = REL; };",
            {"FGPRH": [50.0, 100.0, 200.0]},
            ["5.0", "10.0", "20.0"],
            id="error mode rel scales uncertainty with value",
        ),
        pytest.param(
            """\
            HISTORY_OBSERVATION FOPR {
                ERROR = 5.0;
                ERROR_MODE = ABS;
                SEGMENT SEG {
                    START = 1;
                    STOP  = 3;
                    ERROR = 1.0;
                    ERROR_MODE = ABS;
                };
            };
            """,
            {"FOPRH": [100.0] * 5},
            ["5.0", "1.0", "1.0", "5.0", "5.0"],
            id="segment uses its own error within its range",
        ),
    ],
)
def test_that_history_observation_error_is_computed_per_time_step(
    obs_config_content, summary_vectors, expected_errors
):
    setup_refcase_config(obs_config_content, summary_vectors=summary_vectors)

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

    assert result is not None
    [history_change] = result.history_changes
    for declaration, expected_error in zip(
        history_change.summary_obs_declarations, expected_errors, strict=True
    ):
        assert f"ERROR    = {expected_error}" in declaration


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("start", "stop", "expected_warnings", "expected_errors"),
    [
        pytest.param(
            -2,
            1,
            ["Truncating start of segment to 0"],
            ["1.0", "5.0", "5.0"],
            id="start before zero is truncated to zero",
        ),
        pytest.param(
            1,
            10,
            ["Truncating end of segment to 3"],
            ["5.0", "1.0", "1.0"],
            id="stop beyond last report step is truncated",
        ),
        pytest.param(
            2,
            0,
            ["start after stop", "does not contain any time steps"],
            ["5.0", "5.0", "5.0"],
            id="start after stop is truncated to an empty interval",
        ),
    ],
)
def test_that_segment_bounds_are_truncated_to_valid_range(
    recwarn, start, stop, expected_warnings, expected_errors
):
    setup_refcase_config(
        f"""\
        HISTORY_OBSERVATION FOPR {{
            ERROR = 5.0;
            ERROR_MODE = ABS;
            SEGMENT SEG {{
                START = {start};
                STOP  = {stop};
                ERROR = 1.0;
                ERROR_MODE = ABS;
            }};
        }};
        """,
        summary_vectors={"FOPRH": [100.0] * 3},
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

    warning_messages = [str(w.message) for w in recwarn.list]
    for expected_warning in expected_warnings:
        assert any(expected_warning in m for m in warning_messages)
    assert result is not None
    [history_change] = result.history_changes
    for declaration, expected_error in zip(
        history_change.summary_obs_declarations, expected_errors, strict=True
    ):
        assert f"ERROR    = {expected_error}" in declaration


@pytest.mark.usefixtures("use_tmpdir")
def test_that_general_observation_with_index_file_is_converted_to_use_restart():
    setup_timemap_config(
        """\
        GENERAL_OBSERVATION GEN_OBS {
            DATA = GEN_DATA;
            INDEX_FILE = obs_idx.txt;
            DATE = 2020-01-11;
            OBS_FILE = obs_data.txt;
        };
        """,
        time_map_content="2020-01-01\n2020-01-11\n2020-01-21\n",
        extra_config_lines="GEN_DATA GEN_DATA RESULT_FILE:gen%d.txt REPORT_STEPS:0,1,2",
    )
    Path("obs_idx.txt").write_text("0\n1\n", encoding="utf-8")
    Path("obs_data.txt").write_text("1.0 0.1\n2.0 0.1\n", encoding="utf-8")

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

    assert result is not None
    [gen_obs_change] = result.general_obs_changes
    index_file_line = next(
        line for line in gen_obs_change.declaration.splitlines() if "INDEX_FILE" in line
    )
    assert index_file_line.strip().endswith("obs_idx.txt;")
    assert "RESTART    = 1" in gen_obs_change.declaration


@pytest.mark.usefixtures("use_tmpdir")
def test_that_general_observation_with_value_and_error_is_converted_without_obs_file():
    setup_timemap_config(
        """\
        GENERAL_OBSERVATION GEN_OBS {
            DATA  = GEN_DATA;
            DATE  = 2020-01-11;
            VALUE = 1.0;
            ERROR = 0.1;
        };
        """,
        time_map_content="2020-01-01\n2020-01-11\n2020-01-21\n",
        extra_config_lines="GEN_DATA GEN_DATA RESULT_FILE:gen%d.txt REPORT_STEPS:0,1,2",
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

    assert result is not None
    [gen_obs_change] = result.general_obs_changes
    assert "VALUE      = 1.0" in gen_obs_change.declaration
    assert "ERROR      = 0.1" in gen_obs_change.declaration
    assert "OBS_FILE" not in gen_obs_change.declaration


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("trigger_field", "time_map_content", "expected_date"),
    [
        pytest.param(
            "DAYS  = 10;",
            "2020-01-01\n2020-01-11\n2020-01-21\n",
            datetime(2020, 1, 11),  # ruff: ignore[call-datetime-without-tzinfo]
            id="days",
        ),
        pytest.param(
            "HOURS = 10;",
            "2020-01-01T00:00:00\n2020-01-01T10:00:00\n2020-01-01T20:00:00\n",
            datetime(2020, 1, 1, 10),  # ruff: ignore[call-datetime-without-tzinfo]
            id="hours",
        ),
    ],
)
def test_that_summary_observation_uses_restart_computed_from_time_map(
    trigger_field, time_map_content, expected_date
):
    setup_timemap_config(
        f"""\
        SUMMARY_OBSERVATION FOPR_OBS {{
            VALUE = 110.0;
            ERROR = 5.0;
            {trigger_field}
            KEY   = FOPR;
        }};
        """,
        time_map_content=time_map_content,
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

    assert result is not None
    [summary_change] = result.summary_obs_changes
    assert summary_change.date == expected_date


@pytest.mark.usefixtures("use_tmpdir")
def test_that_missing_obs_config_returns_no_changes():
    Path("config.ert").write_text(
        "NUM_REALIZATIONS 1\n",
        encoding="utf-8",
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

    assert result is None


@pytest.mark.usefixtures("use_tmpdir")
def test_that_unreadable_time_map_file_raises_config_validation_error():
    setup_timemap_config(
        "SUMMARY_OBSERVATION FOPR_OBS { RESTART = 0; };",
        time_map_content="not-a-valid-date\n",
    )

    with pytest.raises(ConfigValidationError, match="Could not read timemap file"):
        remove_refcase_and_time_map_dependence_from_obs_config("config.ert")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_time_map_dates_in_ddmmyyyy_format_are_accepted_with_deprecation_warning(
    caplog,
):
    setup_timemap_config(
        """\
        SUMMARY_OBSERVATION FOPR_OBS {
            VALUE = 0.1;
            ERROR = 0.05;
            DAYS  = 10;
            KEY   = FOPR;
        };
        """,
        time_map_content="01/01/2020\n11/01/2020\n",
    )

    with caplog.at_level("WARNING"):
        result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

    assert "DD/MM/YYYY date format is deprecated" in caplog.text
    assert result is not None
    [summary_change] = result.summary_obs_changes
    assert summary_change.date == datetime(2020, 1, 11)  # ruff: ignore[call-datetime-without-tzinfo]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_history_observation_segment_error_min_sets_error_floor():
    setup_refcase_config(
        """\
        HISTORY_OBSERVATION FOPR {
            ERROR = 5.0;
            ERROR_MODE = ABS;
            SEGMENT SEG {
                START = 0;
                STOP  = 1;
                ERROR = 0.1;
                ERROR_MIN = 2.0;
                ERROR_MODE = RELMIN;
            };
        };
        """,
        summary_vectors={"FOPRH": [1.0]},
    )

    result = remove_refcase_and_time_map_dependence_from_obs_config("config.ert")

    assert result is not None
    [history_change] = result.history_changes
    # The segment's relative error (1.0 * 0.1 = 0.1) is below ERROR_MIN, so the
    # minimum of 2.0 is used instead.
    assert len(history_change.summary_obs_declarations) == 1
    assert "ERROR    = 2.0" in history_change.summary_obs_declarations[0]
