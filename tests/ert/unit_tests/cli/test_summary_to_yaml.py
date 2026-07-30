import logging
import warnings
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import numpy as np
import pytest
from ruamel.yaml import YAML

from ert.cli.main import ErtCliError
from ert.config._observations import ErrorModes, SummaryObservation
from ert.observation_converters import convert_observations
from ert.observation_converters.summary_to_yaml import YamlConverter
from ert.plugins import ErtRuntimePlugins
from tests.ert.unit_tests.cli.test_summary_to_bulk import _make_breakthrough_obs


def _make_summary_obs(
    key: str = "WOPR",
    well: str = "",
    date: str = "2010-01-27",
    shape_id: int | None = None,
) -> SummaryObservation:
    key += f":{well}" if well else ""
    return SummaryObservation(
        name="foo",
        key=key,
        value=0.5,
        error=0.02,
        date=date,
        shape_id=shape_id,
    )


@pytest.mark.usefixtures("snake_oil_case")
@pytest.mark.slow
def test_that_happy_path_on_snake_oil_produces_yaml_and_stdout(capsys):
    args = MagicMock(format="yaml", config="snake_oil.ert")
    convert_observations(args, ErtRuntimePlugins())

    expected_yaml_content = dedent("""\
    smry:
    - key: WOPR:OP1
      observations:
      - date: '2010-03-31'
        value: 0.1
        error: 0.05
      - date: '2010-12-26'
        value: 0.7
        error: 0.07
      - date: '2011-12-21'
        value: 0.5
        error: 0.05
      - date: '2012-12-15'
        value: 0.3
        error: 0.075
      - date: '2013-12-10'
        value: 0.2
        error: 0.035
      - date: '2015-03-15'
        value: 0.015
        error: 0.01
    """)
    yaml_content = Path(YamlConverter.TARGET_FILE).read_text(encoding="utf-8")
    assert yaml_content == expected_yaml_content

    expected_stdout = (
        f"Successfully wrote summary observations to '{YamlConverter.TARGET_FILE}'."
    )
    stdout = capsys.readouterr().out
    assert expected_stdout in stdout


def test_that_empty_observations_raises_ert_cli_error():
    with pytest.raises(ErtCliError, match="No summary observations in configuration"):
        YamlConverter([])


def test_that_no_summary_observations_raises_ert_cli_error():
    brt_obs = _make_breakthrough_obs(well="OP1")
    with pytest.raises(ErtCliError, match="No summary observations in configuration"):
        YamlConverter([brt_obs])


def test_that_observations_with_same_summary_key_are_gathered_in_yaml_dict(use_tmpdir):
    k1, k2 = "bar", "foo"
    observations = 2 * [
        _make_summary_obs(key=k1),
        _make_summary_obs(key=k2),
    ]

    converter = YamlConverter(observations=observations)
    summary_dicts = converter._summary_to_yaml_dict()

    keys = [entry["key"] for entry in summary_dicts["smry"]]
    assert keys == [k1, k2]

    observations = [s_d["observations"] for s_d in summary_dicts["smry"]]
    assert all(len(o) == 2 for o in observations)


def test_that_observations_with_different_summary_keys_are_separated_in_yaml_dict():
    k1, k2 = "bar", "foo"
    observations = [
        _make_summary_obs(key=k1),
        _make_summary_obs(key=k2),
    ]

    converter = YamlConverter(observations=observations)
    result = converter._summary_to_yaml_dict()

    keys = [entry["key"] for entry in result["smry"]]
    assert keys == [k1, k2]


def test_that_dumping_to_yaml_is_skipped_when_file_already_exists(use_tmpdir):
    observations = [_make_summary_obs()]

    Path(YamlConverter.TARGET_FILE).write_text("existing", encoding="utf-8")
    assert Path(YamlConverter.TARGET_FILE).is_file()

    converter = YamlConverter(observations=observations)
    with pytest.raises(
        ErtCliError,
        match=(
            rf"A file with name '{YamlConverter.TARGET_FILE}' already exists. "
            "Will not overwrite it and exit instead."
        ),
    ):
        converter.export_yaml()


def test_that_config_warnings_are_caught_instead_of_printed_to_terminal(
    caplog, use_tmpdir
):
    caplog.set_level(logging.INFO)
    config = "config.ert"
    obs_config = "obs.txt"
    # This setup expects the warning:
    # 'Config contains a SUMMARY key but no forward model steps'
    # to be raised
    config_content = f"""\
    NUM_REALIZATIONS 5
    SUMMARY *
    ECLBASE FOO
    OBS_CONFIG {obs_config}
    """
    obs_config_content = """\
    SUMMARY_OBSERVATION {
        KEY=FOPR;
        VALUE=10;
        ERROR=5;
        DATE=2010-10-10;
    };"""
    Path(config).write_text(config_content, encoding="utf-8")
    Path(obs_config).write_text(obs_config_content, encoding="utf-8")

    args = MagicMock(format="yaml", config=config)
    with warnings.catch_warnings(record=True) as w:
        convert_observations(args, ErtRuntimePlugins())
    assert len(w) == 0


def test_that_yaml_converter_sorts_observations_by_date(use_tmpdir):
    d1 = "2000-01-01"
    d4 = "2000-01-11"
    d3 = "2000-01-03"
    d5 = "2000-02-11"
    d2 = "2000-01-02"

    unsorted_obs = [_make_summary_obs(date=d) for d in [d1, d4, d3, d5, d2]]
    yaml_dict = YamlConverter(unsorted_obs)._summary_to_yaml_dict()

    obs_dicts = yaml_dict["smry"][0]["observations"]
    yaml_key_order = [obs_dict["date"] for obs_dict in obs_dicts]
    assert yaml_key_order == [d1, d2, d3, d4, d5]


def test_that_yaml_converter_natsorts_summary_keys(use_tmpdir):
    k1 = "WOPR:OP1"
    k3 = "WOPR:OP13"
    k2 = "WOPR:OP2"
    unsorted_obs = [
        _make_summary_obs(key=k1),
        _make_summary_obs(key=k3),
        _make_summary_obs(key=k2),
    ]
    yaml_dict = YamlConverter(unsorted_obs)._summary_to_yaml_dict()

    yaml_key_order = [d["key"] for d in yaml_dict["smry"]]
    assert yaml_key_order == [k1, k2, k3]


def test_that_yaml_converter_retains_time_precision_when_present():
    date_precision = "2000-01-01"
    second_precision = date_precision + "T00:00:01"
    minute_precision = date_precision + "T00:01:00"
    hour_precision = date_precision + "T01:00:00"
    obs = [
        _make_summary_obs(date=d)
        for d in [date_precision, hour_precision, minute_precision, second_precision]
    ]

    yaml_dict = YamlConverter(obs)._summary_to_yaml_dict()

    yaml_obs = yaml_dict["smry"][0]["observations"]
    assert [o["date"] for o in yaml_obs] == [
        date_precision,
        second_precision,
        minute_precision,
        hour_precision,
    ]


_VALUE = 10
_ERROR = 5


def _setup_and_convert_with_error_mode_config(
    error_mode: ErrorModes, error_min=None
) -> int | float:
    config = "config.ert"
    obs_config = "obs.txt"

    config_content = f"""\
        NUM_REALIZATIONS 5
        ECLBASE FOO
        OBS_CONFIG {obs_config}
        """
    error_min_line = f"ERROR_MIN={error_min};" if error_min is not None else ""
    obs_config_content = f"""\
        SUMMARY_OBSERVATION {{
            KEY=FOPR;
            VALUE={_VALUE};
            ERROR={_ERROR};
            ERROR_MODE={error_mode};
            DATE=2010-10-10;
            {error_min_line}
        }};"""
    Path(config).write_text(config_content, encoding="utf-8")
    Path(obs_config).write_text(obs_config_content, encoding="utf-8")

    args = MagicMock(format="yaml", config=config)
    convert_observations(args, ErtRuntimePlugins())

    yaml = YAML()
    with Path.open(YamlConverter.TARGET_FILE) as f:
        yaml_output = yaml.load(f)

    return yaml_output["smry"][0]["observations"][0]["error"]


def test_that_absolute_error_is_unchanged_after_conversion(use_tmpdir):
    error = _setup_and_convert_with_error_mode_config(error_mode=ErrorModes.ABS)
    assert error == _ERROR


def test_that_relative_error_is_product_of_val_and_error_after_conversion(use_tmpdir):
    error = _setup_and_convert_with_error_mode_config(error_mode=ErrorModes.REL)
    assert error == _ERROR * _VALUE


def test_that_rel_min_error_is_product_of_val_and_err_after_conversion_given_no_err_min(
    use_tmpdir,
):
    error = _setup_and_convert_with_error_mode_config(error_mode=ErrorModes.RELMIN)
    assert error == _ERROR * _VALUE


def test_that_relative_min_error_is_unchanged_after_conversion_given_error_min_key(
    use_tmpdir,
):
    error_min = np.inf
    error = _setup_and_convert_with_error_mode_config(
        error_mode=ErrorModes.RELMIN, error_min=error_min
    )
    assert error == error_min


def test_that_errors_are_formatted_to_user_with_message(use_tmpdir):
    obs_config = "foo"
    summary_obs = "SUMMARY_OBSERVATION { This is not a valid observation };"
    Path(obs_config).write_text(
        summary_obs,
        encoding="utf-8",
    )

    ert_config = "config.ert"
    minimal_ert_config = f"""\
    NUM_REALIZATIONS 10
    ECLBASE foo
    OBS_CONFIG {obs_config}
    """
    Path(ert_config).write_text(minimal_ert_config, encoding="utf-8")
    args = MagicMock(format="yaml", config=ert_config)
    with pytest.raises(ErtCliError, match="Failed to internalize the ert config"):
        convert_observations(args, ErtRuntimePlugins())
