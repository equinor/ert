import logging
import os
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

from everest import ConfigKeys as CK
from everest import MetaDataColumnNames as MDCN
from everest.bin.everexport_script import everexport_entry
from everest.bin.utils import ProgressBar
from everest.config import EverestConfig
from tests.everest.utils import (
    create_cached_mocked_test_case,
    satisfy,
    satisfy_callable,
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


def validate_export_mock(config):
    return ([], True)


@pytest.fixture()
def cache_dir(request, monkeypatch):
    return create_cached_mocked_test_case(request, monkeypatch)


@patch("everest.bin.utils.export_with_progress", side_effect=export_mock)
def test_everexport_entry_run(mocked_func, copy_math_func_test_data_to_tmp):
    """Test running everexport with not flags"""
    # NOTE: there is probably a bug concerning output folders. Everexport
    # seems to assume that the folder where the file will be saved exists.
    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    export_file_path = config.export_path
    assert not os.path.isfile(export_file_path)

    everexport_entry([CONFIG_FILE_MINIMAL])

    assert os.path.isfile(export_file_path)
    df = pd.read_csv(export_file_path, sep=";")
    assert df.equals(TEST_DATA)


@patch("everest.bin.utils.export_with_progress", side_effect=empty_mock)
def test_everexport_entry_empty(mocked_func, copy_math_func_test_data_to_tmp):
    """Test running everexport with no data"""
    # NOTE: When there is no data (ie, the optimization has not yet run)
    # the current behavior is to create an empty .csv file. It is arguable
    # whether that is really the desired behavior, but for now we assume
    # it is and we test against that expected behavior.
    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    export_file_path = config.export_path
    assert not os.path.isfile(export_file_path)

    everexport_entry([CONFIG_FILE_MINIMAL])

    assert os.path.isfile(export_file_path)
    with open(export_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert not content.strip()


@patch(
    "everest.bin.everexport_script.validate_export",
    side_effect=validate_export_mock,
)
@patch("everest.bin.utils.export")
@pytest.mark.fails_on_macos_github_workflow
def test_everexport_entry_batches(
    mocked_func, validate_export_mock, copy_math_func_test_data_to_tmp
):
    """Test running everexport with the --batches flag"""
    everexport_entry([CONFIG_FILE_MINIMAL, "--batches", "0", "2"])

    def check_export_batches(config: EverestConfig):
        batches = (
            config.export.batches if config.export is not None else None
        ) or False
        return set(batches) == {0, 2}

    if ProgressBar:  # different calls if ProgressBar available or not
        mocked_func.assert_called_once_with(
            config=satisfy(check_export_batches),
            export_ecl=True,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once_with(config=satisfy(check_export_batches))


@patch("everest.bin.everexport_script.export_to_csv")
def test_everexport_entry_no_export(mocked_func, copy_math_func_test_data_to_tmp):
    """Test running everexport on config file with skip_export flag
    set to true"""

    # Add export section to config file and set run_export flag to false
    with open(CONFIG_FILE_MINIMAL, "a", encoding="utf-8") as f:
        f.write(
            "{export}:\n  {skip}: True".format(export=CK.EXPORT, skip=CK.SKIP_EXPORT)
        )

    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    export_file_path = config.export_path
    assert not os.path.isfile(export_file_path)

    everexport_entry([CONFIG_FILE_MINIMAL])
    # Check export to csv is called even if the skip_export entry is in the
    # config file
    mocked_func.assert_called_once()


@patch("everest.bin.everexport_script.export_to_csv")
def test_everexport_entry_empty_export(mocked_func, copy_math_func_test_data_to_tmp):
    """Test running everexport on config file with empty export section"""

    # Add empty export section to config file
    with open(CONFIG_FILE_MINIMAL, "a", encoding="utf-8") as f:
        f.write(f"{CK.EXPORT}:\n")

    everexport_entry([CONFIG_FILE_MINIMAL])
    # Check export to csv is called even if export section is empty
    mocked_func.assert_called_once()


@patch("everest.bin.utils.export")
@pytest.mark.fails_on_macos_github_workflow
def test_everexport_entry_no_usr_def_ecl_keys(
    mocked_func, copy_mocked_test_data_to_tmp
):
    """Test running everexport with config file containing only the
    keywords label without any list of keys"""

    # Add export section to config file and set run_export flag to false
    with open(CONFIG_FILE_MOCKED_TEST_CASE, "a", encoding="utf-8") as f:
        f.write(
            "{export}:\n  {keywords}:".format(
                export=CK.EXPORT,
                keywords=CK.KEYWORDS,
            )
        )

    everexport_entry([CONFIG_FILE_MOCKED_TEST_CASE])

    def condition(config: EverestConfig):
        batches = config.export.batches if config.export is not None else None
        keys = config.export.keywords if config.export is not None else None
        return batches is None and keys is None

    if ProgressBar:
        mocked_func.assert_called_once_with(
            config=satisfy(condition),
            export_ecl=True,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once_with(config=satisfy(condition), export_ecl=True)


@patch("everest.bin.utils.export")
@pytest.mark.fails_on_macos_github_workflow
def test_everexport_entry_internalized_usr_def_ecl_keys(
    mocked_func, cache_dir, copy_mocked_test_data_to_tmp
):
    """Test running everexport with config file containing a key in the
    list of user defined ecl keywords, that has been internalized on
    a previous run"""

    shutil.copytree(
        cache_dir / "mocked_multi_batch_output",
        "mocked_multi_batch_output",
        dirs_exist_ok=True,
    )

    user_def_keys = ["FOPT"]

    # Add export section to config file and set run_export flag to false
    with open(CONFIG_FILE_MOCKED_TEST_CASE, "a", encoding="utf-8") as f:
        f.write(
            "{export}:\n  {keywords}: {keys}".format(
                export=CK.EXPORT, keywords=CK.KEYWORDS, keys=user_def_keys
            )
        )

    everexport_entry([CONFIG_FILE_MOCKED_TEST_CASE])

    def condition(config: EverestConfig):
        batches = config.export.batches if config.export is not None else None
        keys = config.export.keywords if config.export is not None else None

        return batches is None and keys == user_def_keys

    if ProgressBar:
        mocked_func.assert_called_once_with(
            config=satisfy(condition),
            export_ecl=True,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once_with(config=satisfy(condition), export_ecl=True)


@patch("everest.bin.utils.export")
@pytest.mark.fails_on_macos_github_workflow
def test_everexport_entry_non_int_usr_def_ecl_keys(
    mocked_func, cache_dir, caplog, copy_mocked_test_data_to_tmp
):
    """Test running everexport  when config file contains non internalized
    ecl keys in the user defined keywords list"""

    shutil.copytree(
        cache_dir / "mocked_multi_batch_output",
        "mocked_multi_batch_output",
        dirs_exist_ok=True,
    )

    non_internalized_key = "KEY"
    user_def_keys = ["FOPT", non_internalized_key]

    # Add export section to config file and set run_export flag to false
    with open(CONFIG_FILE_MOCKED_TEST_CASE, "a", encoding="utf-8") as f:
        f.write(
            "{export}:\n  {keywords}: {keys}".format(
                export=CK.EXPORT, keywords=CK.KEYWORDS, keys=user_def_keys
            )
        )

    with caplog.at_level(logging.DEBUG):
        everexport_entry([CONFIG_FILE_MOCKED_TEST_CASE])

    assert (
        f"Non-internalized ecl keys selected for export '{non_internalized_key}'"
        in "\n".join(caplog.messages)
    )

    def condition(config: EverestConfig):
        batches = config.export.batches if config.export is not None else None
        keys = config.export.keywords if config.export is not None else None

        return batches is None and keys == user_def_keys

    if ProgressBar:
        mocked_func.assert_called_once_with(
            config=satisfy(condition),
            export_ecl=False,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once_with(config=satisfy(condition), export_ecl=False)


@patch("everest.bin.utils.export")
@pytest.mark.fails_on_macos_github_workflow
def test_everexport_entry_not_available_batches(
    mocked_func, cache_dir, caplog, copy_mocked_test_data_to_tmp
):
    """Test running everexport  when config file contains non existing
    batch numbers in the list of user defined batches"""

    shutil.copytree(
        cache_dir / "mocked_multi_batch_output",
        "mocked_multi_batch_output",
        dirs_exist_ok=True,
    )

    na_batch = 42
    user_def_batches = [0, na_batch]
    mocked_test_config_file = "mocked_multi_batch.yml"

    # Add export section to config file and set run_export flag to false
    with open(mocked_test_config_file, "a", encoding="utf-8") as f:
        f.write(
            "{export}:\n  {batch_key}: {batches}".format(
                export=CK.EXPORT, batch_key=CK.BATCHES, batches=user_def_batches
            )
        )

    with caplog.at_level(logging.DEBUG):
        everexport_entry([mocked_test_config_file])

    assert (
        f"Batch {na_batch} not found in optimization results."
        f" Skipping for current export" in "\n".join(caplog.messages)
    )

    def condition(config: EverestConfig):
        batches = config.export.batches if config.export is not None else None
        keys = config.export.keywords if config.export is not None else None
        return batches == [0] and keys is None

    if ProgressBar:
        mocked_func.assert_called_once_with(
            config=satisfy(condition),
            export_ecl=True,
            progress_callback=satisfy_callable(),
        )
    else:
        mocked_func.assert_called_once_with(config=satisfy(condition), export_ecl=True)
