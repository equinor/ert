from pathlib import Path
from textwrap import dedent

import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert.config import ConfigValidationError, ErtConfig
from ert.config.parsing import ConfigKeys


def test_default_num_cpu():
    ert_config = ErtConfig.from_file_contents("NUM_REALIZATIONS 1")
    assert ert_config.queue_config.preferred_num_cpu == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_from_config_preferred():
    data_file = "dfile"
    config_num_cpu = 17
    data_file_num_cpu = 4
    Path(data_file).write_text(
        dedent(f"""\
            PARALLEL
            {data_file_num_cpu} DISTRIBUTED/
        """),
        encoding="utf-8",
    )
    assert (
        ErtConfig.from_dict(
            {
                ConfigKeys.NUM_CPU: config_num_cpu,
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.DATA_FILE: data_file,
            }
        ).queue_config.preferred_num_cpu
        == config_num_cpu
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@given(st.text())
@pytest.mark.usefixtures("use_tmpdir")
def test_reading_num_cpu_from_data_file_does_not_crash(data_file_contents):
    data_file = "case.data"
    Path(data_file).write_text(data_file_contents, encoding="utf-8")
    _ = ErtConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 1,
            ConfigKeys.DATA_FILE: data_file,
        }
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@given(st.binary())
@pytest.mark.usefixtures("use_tmpdir")
def test_reading_num_cpu_from_binary_data_file_does_not_crash(data_file_contents):
    data_file = "case.data"
    Path(data_file).write_bytes(data_file_contents)
    _ = ErtConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 1,
            ConfigKeys.DATA_FILE: data_file,
        }
    )


@pytest.mark.parametrize(
    "parallelsuffix", [("/"), (" /"), (" DISTRIBUTED/"), (" DISTRIBUTED /")]
)
@pytest.mark.parametrize(
    "casetitle",
    [
        "CASE",
        "-- A CASE --",
        "PARALLEL Tutorial Case",
        "",  # Not valid input in some reservoir simulators
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_from_data_file_used_if_config_num_cpu_not_set(
    parallelsuffix, casetitle
):
    data_file_num_cpu = 4
    data_file = "case.data"
    Path(data_file).write_text(
        dedent(
            f"""\
        RUNSPEC
        --comment
        TITLE
        {casetitle}
        PARALLEL
         {data_file_num_cpu}{parallelsuffix}
    """,
        ),
        encoding="utf-8",
    )

    assert (
        ErtConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.DATA_FILE: data_file,
            }
        ).queue_config.preferred_num_cpu
        == data_file_num_cpu
    )


@pytest.mark.parametrize(
    "num_cpu_value, error_msg",
    [
        (-1, "must have a positive integer value as argument"),
        (0, "must have a positive integer value as argument"),
        (1.5, "must have an integer value as argument"),
    ],
)
def test_wrong_num_cpu_raises_validation_error(num_cpu_value, error_msg):
    with pytest.raises(ConfigValidationError, match=error_msg):
        ErtConfig.from_file_contents(
            f"{ConfigKeys.NUM_REALIZATIONS} 1\n{ConfigKeys.NUM_CPU} {num_cpu_value}\n"
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_can_be_parsed_from_run_template():
    num_cpu = 4
    template_file = "templatecase.data"
    Path(template_file).write_text(
        dedent(
            f"""\
        PARALLEL
         {num_cpu} /
        """
        ),
        encoding="utf-8",
    )
    assert (
        ErtConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.RUN_TEMPLATE: [[template_file, "<ECLBASE>.DATA"]],
            }
        ).queue_config.preferred_num_cpu
        == num_cpu
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_explicit_num_cpu_overrides_run_template():
    num_cpu = 4
    template_file = "templatecase.data"
    Path(template_file).write_text(
        dedent(
            f"""\
        PARALLEL
         {num_cpu} /
        """
        ),
        encoding="utf-8",
    )
    assert (
        ErtConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.RUN_TEMPLATE: [[template_file, "<ECLBASE>.DATA"]],
                ConfigKeys.NUM_CPU: 5,
            }
        ).queue_config.preferred_num_cpu
        == 5
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_can_be_parsed_from_run_template_with_eclbase_set():
    num_cpu = 4
    data_file = "templatecase.data"
    Path(data_file).write_text(
        dedent(
            f"""\
        PARALLEL
         {num_cpu} /
        """
        ),
        encoding="utf-8",
    )
    assert (
        ErtConfig.from_dict(
            {
                ConfigKeys.ECLBASE: "FOO.DATA",
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.RUN_TEMPLATE: [[data_file, "FOO.DATA"]],
            }
        ).queue_config.preferred_num_cpu
        == num_cpu
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_is_not_parsed_from_other_than_eclbase():
    fallback_num_cpu = 1
    num_cpu = 4
    template_file = "templatecase.data"
    Path(template_file).write_text(
        dedent(
            f"""\
        PARALLEL
         {num_cpu} /
        """
        ),
        encoding="utf-8",
    )
    assert (
        ErtConfig.from_dict(
            {
                ConfigKeys.ECLBASE: "BAR.DATA",
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.RUN_TEMPLATE: [[template_file, "FOO.DATA"]],
            }
        ).queue_config.preferred_num_cpu
        == fallback_num_cpu
    )
