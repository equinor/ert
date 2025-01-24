import logging
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from everest import MetaDataColumnNames as MDCN
from everest.bin.everexport_script import everexport_entry
from everest.bin.utils import ProgressBar
from everest.config import EverestConfig, ExportConfig
from tests.everest.utils import (
    satisfy,
    satisfy_callable,
    satisfy_type,
)

CONFIG_FILE_MINIMAL = "config_minimal.yml"

CONFIG_FILE_MOCKED_TEST_CASE = "mocked_multi_batch.yml"


pytestmark = pytest.mark.xdist_group(name="starts_everest")

TEST_DATA = pd.DataFrame(
    columns=[
        MDCN.BATCH,
        MDCN.SIMULATION,
        MDCN.IS_GRADIENT,
        MDCN.START_TIME,
    ],
    data=[
        [0, 0, False, 0.0],  # First func evaluation on 2 realizations
        [0, 1, False, 0.0],
        [0, 2, True, 0.0],  # First grad evaluation 2 perts per real
        [0, 3, True, 0.1],
        [0, 4, True, 0.1],
        [0, 5, True, 0.1],
        [1, 0, False, 0.3],
        [1, 1, False, 0.32],
        [2, 0, True, 0.5],
        [2, 1, True, 0.5],
        [2, 2, True, 0.5],
        [2, 3, True, 0.6],
    ],
)


def export_mock(config, export_ecl=True, progress_callback=lambda _: None):
    progress_callback(1.0)
    return TEST_DATA


def empty_mock(config, export_ecl=True, progress_callback=lambda _: None):
    progress_callback(1.0)
    return pd.DataFrame()


def validate_export_mock(**_):
    return ([], True)


@patch("everest.bin.everexport_script.export_with_progress", side_effect=export_mock)
def test_everexport_entry_run(_, cached_example):
    """Test running everexport with not flags"""
    config_path, config_file, _ = cached_example("math_func/config_minimal.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)
    export_file_path = config.export_path
    assert not os.path.isfile(export_file_path)

    everexport_entry([config_file])

    assert os.path.isfile(export_file_path)
    df = pd.read_csv(export_file_path, sep=";")
    assert df.equals(TEST_DATA)


@patch("everest.bin.everexport_script.export_with_progress", side_effect=empty_mock)
def test_everexport_entry_empty(mocked_func, cached_example):
    """Test running everexport with no data"""
    # NOTE: When there is no data (ie, the optimization has not yet run)
    # the current behavior is to create an empty .csv file. It is arguable
    # whether that is really the desired behavior, but for now we assume
    # it is and we test against that expected behavior.
    config_path, config_file, _ = cached_example("math_func/config_minimal.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)
    export_file_path = config.export_path
    assert not os.path.isfile(export_file_path)

    everexport_entry([CONFIG_FILE_MINIMAL])

    assert os.path.isfile(export_file_path)
    with open(export_file_path, encoding="utf-8") as f:
        content = f.read()
    assert not content.strip()


@patch(
    "everest.bin.everexport_script.check_for_errors",
    side_effect=validate_export_mock,
)
@patch("everest.bin.utils.export_data")
@pytest.mark.skip_mac_ci
def test_everexport_entry_batches(mocked_func, validate_export_mock, cached_example):
    """Test running everexport with the --batches flag"""
    _, config_file, _ = cached_example("math_func/config_minimal.yml")
    everexport_entry([config_file, "--batches", "0", "2"])

    def check_export_batches(config):
        batches = (config.batches if config is not None else None) or False
        return set(batches) == {0, 2}

    if ProgressBar:  # different calls if ProgressBar available or not
        mocked_func.assert_called_once_with(
            export_config=satisfy(check_export_batches),
            output_dir=satisfy_type(str),
            data_file=None,
            export_ecl=True,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once()


@patch("everest.bin.everexport_script.export_to_csv")
def test_everexport_entry_no_export(mocked_func, cached_example):
    """Test running everexport on config file with skip_export flag
    set to true"""

    config_path, config_file, _ = cached_example("math_func/config_minimal.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)
    config.export = ExportConfig(skip_export=True)
    # Add export section to config file and set run_export flag to false
    export_file_path = config.export_path
    assert not os.path.isfile(export_file_path)

    everexport_entry([CONFIG_FILE_MINIMAL])
    # Check export to csv is called even if the skip_export entry is in the
    # config file
    mocked_func.assert_called_once()


@patch("everest.bin.everexport_script.export_to_csv")
def test_everexport_entry_empty_export(mocked_func, cached_example):
    """Test running everexport on config file with empty export section"""
    _, config_file, _ = cached_example("math_func/config_minimal.yml")

    # Add empty export section to config file
    with open(config_file, "a", encoding="utf-8") as f:
        f.write("export:\n")

    everexport_entry([config_file])
    # Check export to csv is called even if export section is empty
    mocked_func.assert_called_once()


@patch("everest.bin.utils.export_data")
@pytest.mark.skip_mac_ci
def test_everexport_entry_no_usr_def_ecl_keys(mocked_func, cached_example):
    """Test running everexport with config file containing only the
    keywords label without any list of keys"""

    _, config_file, _ = cached_example(
        "../../tests/everest/test_data/mocked_test_case/mocked_multi_batch.yml"
    )

    # Add export section to config file and set run_export flag to false
    with open(config_file, "a", encoding="utf-8") as f:
        f.write("export:\n  keywords:")

    everexport_entry([config_file])

    def condition(config):
        batches = config.batches if config is not None else None
        keys = config.keywords if config is not None else None

        return batches is None and keys is None

    if ProgressBar:
        mocked_func.assert_called_once_with(
            export_config=satisfy(condition),
            output_dir=satisfy_type(str),
            data_file=satisfy_type(str),
            export_ecl=True,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once()


@patch("everest.bin.utils.export_data")
@pytest.mark.skip_mac_ci
def test_everexport_entry_internalized_usr_def_ecl_keys(mocked_func, cached_example):
    """Test running everexport with config file containing a key in the
    list of user defined ecl keywords, that has been internalized on
    a previous run"""

    _, config_file, _ = cached_example(
        "../../tests/everest/test_data/mocked_test_case/mocked_multi_batch.yml"
    )
    user_def_keys = ["FOPT"]

    # Add export section to config file and set run_export flag to false
    with open(config_file, "a", encoding="utf-8") as f:
        f.write(f"export:\n  keywords: {user_def_keys}")

    everexport_entry([config_file])

    def condition(config):
        batches = config.batches if config is not None else None
        keys = config.keywords if config is not None else None

        return batches is None and keys == user_def_keys

    if ProgressBar:
        mocked_func.assert_called_once_with(
            export_config=satisfy(condition),
            output_dir=satisfy_type(str),
            data_file=satisfy_type(str),
            export_ecl=True,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once()


@patch("everest.bin.utils.export_data")
@pytest.mark.skip_mac_ci
def test_everexport_entry_non_int_usr_def_ecl_keys(mocked_func, caplog, cached_example):
    """Test running everexport  when config file contains non internalized
    ecl keys in the user defined keywords list"""

    _, config_file, _ = cached_example(
        "../../tests/everest/test_data/mocked_test_case/mocked_multi_batch.yml"
    )

    non_internalized_key = "KEY"
    user_def_keys = ["FOPT", non_internalized_key]

    # Add export section to config file and set run_export flag to false
    with open(config_file, "a", encoding="utf-8") as f:
        f.write(f"export:\n  keywords: {user_def_keys}")

    with caplog.at_level(logging.DEBUG):
        everexport_entry([config_file])

    assert (
        f"Non-internalized ecl keys selected for export '{non_internalized_key}'"
        in "\n".join(caplog.messages)
    )

    def condition(config):
        batches = config.batches if config is not None else None
        keys = config.keywords if config is not None else None

        return batches is None and keys == user_def_keys

    if ProgressBar:
        mocked_func.assert_called_once_with(
            export_config=satisfy(condition),
            output_dir=satisfy_type(str),
            data_file=satisfy_type(str),
            export_ecl=False,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once()


@patch("everest.bin.utils.export_data")
@pytest.mark.skip_mac_ci
def test_everexport_entry_not_available_batches(mocked_func, caplog, cached_example):
    """Test running everexport  when config file contains non existing
    batch numbers in the list of user defined batches"""

    _, config_file, _ = cached_example(
        "../../tests/everest/test_data/mocked_test_case/mocked_multi_batch.yml"
    )

    na_batch = 42
    user_def_batches = [0, na_batch]

    # Add export section to config file and set run_export flag to false
    with open(config_file, "a", encoding="utf-8") as f:
        f.write(f"export:\n  {'batches'}: {user_def_batches}")

    with caplog.at_level(logging.DEBUG):
        everexport_entry([config_file])

    assert (
        f"Batch {na_batch} not found in optimization results."
        f" Skipping for current export" in "\n".join(caplog.messages)
    )

    def condition(config):
        batches = config.batches if config is not None else None
        keys = config.keywords if config is not None else None
        return batches == [0] and keys is None

    if ProgressBar:
        mocked_func.assert_called_once_with(
            export_config=satisfy(condition),
            output_dir=satisfy_type(str),
            data_file=satisfy_type(str),
            export_ecl=True,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once()
