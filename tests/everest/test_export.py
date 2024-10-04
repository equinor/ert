import os
import shutil

import pandas as pd
import pytest

from everest import filter_data
from everest.bin.utils import export_with_progress
from everest.config import EverestConfig
from everest.config.export_config import ExportConfig
from everest.export import export, validate_export
from tests.everest.utils import create_cached_mocked_test_case, relpath

CONFIG_FILE_MOCKED_TEST_CASE = "mocked_multi_batch.yml"
CASHED_RESULTS_FOLDER = relpath("test_data", "cached_results_config_multiobj")
CONFIG_FILE = "config_multiobj.yml"
DATA = pd.DataFrame(
    {
        "WOPT:WELL0": range(4),
        "MONKEY": 4 * [0],
        "WCON:WELL1": 4 * [14],
        "GOPT:GROUP0": [5, 6, 2, 1],
        "WOPT:WELL1": range(4),
    }
)

pytestmark = pytest.mark.xdist_group(name="starts_everest")


@pytest.fixture()
def cache_dir(request, monkeypatch):
    return create_cached_mocked_test_case(request, monkeypatch)


def assertEqualDataFrames(x, y):
    assert set(x.columns) == set(y.columns)
    for col in x.columns:
        assert list(x[col]) == list(y[col])


def test_filter_no_wildcard():
    keywords = ["MONKEY", "Dr. MONKEY", "WOPT:WELL1"]
    assertEqualDataFrames(DATA[["MONKEY", "WOPT:WELL1"]], filter_data(DATA, keywords))


def test_filter_leading_wildcard():
    keywords = ["*:WELL1"]
    assertEqualDataFrames(
        DATA[["WCON:WELL1", "WOPT:WELL1"]], filter_data(DATA, keywords)
    )


def test_filter_trailing_wildcard():
    keywords = ["WOPT:*", "MONKEY"]
    assertEqualDataFrames(
        DATA[["MONKEY", "WOPT:WELL0", "WOPT:WELL1"]],
        filter_data(DATA, keywords),
    )


def test_filter_double_wildcard():
    keywords = ["*OPT:*0"]
    assertEqualDataFrames(
        DATA[["WOPT:WELL0", "GOPT:GROUP0"]], filter_data(DATA, keywords)
    )


def test_export_only_non_gradient_with_increased_merit(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    os.makedirs(config.optimization_output_dir)
    shutil.copy(
        os.path.join(CASHED_RESULTS_FOLDER, "seba.db"),
        os.path.join(config.optimization_output_dir, "seba.db"),
    )

    # Default export functionality when no export section is defined
    df = export(config)

    # Test that the default export functionality generated data frame
    # contains only non gradient simulations
    for grad_flag in df["is_gradient"].values:
        assert grad_flag == 0

    # Test that the default export functionality generated data frame
    # contains only rows with increased merit simulations
    for merit_flag in df["increased_merit"].values:
        assert merit_flag == 1


def test_export_only_non_gradient(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    os.makedirs(config.optimization_output_dir)
    shutil.copy(
        os.path.join(CASHED_RESULTS_FOLDER, "seba.db"),
        os.path.join(config.optimization_output_dir, "seba.db"),
    )

    # Add export section to config
    config.export = ExportConfig(discard_rejected=False)

    df = export(config)

    # Check if only discard rejected key is set to False in the export
    # section the export will contain only non-gradient simulations
    assert 1 not in df["is_gradient"].values

    # Check the export contains both increased merit and non increased merit
    # when discard rejected key is set to False
    assert 0 in df["increased_merit"].values
    assert 1 in df["increased_merit"].values


def test_export_only_increased_merit(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    os.makedirs(config.optimization_output_dir)
    shutil.copy(
        os.path.join(CASHED_RESULTS_FOLDER, "seba.db"),
        os.path.join(config.optimization_output_dir, "seba.db"),
    )

    # Add export section to config
    config.export = ExportConfig(discard_gradient=False)

    df = export(config)

    # Check the export contains both gradient and non-gradient simulation
    # when discard gradient key is set to False
    assert 1 in df["is_gradient"].values
    assert 0 in df["is_gradient"].values

    # Check if only discard gradient key is set to False
    # the export will contain only increased merit simulations
    assert 0 not in df["increased_merit"].values


def test_export_all_batches(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    os.makedirs(config.optimization_output_dir)
    shutil.copy(
        os.path.join(CASHED_RESULTS_FOLDER, "seba.db"),
        os.path.join(config.optimization_output_dir, "seba.db"),
    )

    # Add export section to config
    config.export = ExportConfig(discard_gradient=False, discard_rejected=False)

    df = export(config)

    # Check the export contains both gradient and non-gradient simulation
    assert 1 in df["is_gradient"].values
    assert 0 in df["is_gradient"].values

    # Check the export contains both merit and non-merit simulation
    assert 1 in df["increased_merit"].values
    assert 0 in df["increased_merit"].values


def test_export_only_give_batches(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    os.makedirs(config.optimization_output_dir)
    shutil.copy(
        os.path.join(CASHED_RESULTS_FOLDER, "seba.db"),
        os.path.join(config.optimization_output_dir, "seba.db"),
    )

    # Add export section to config
    config.export = ExportConfig(discard_gradient=True, batches=[2])

    df = export(config)
    # Check only simulations from given batches are present in export
    for id in df["batch"].values:
        assert id == 2


@pytest.mark.fails_on_macos_github_workflow
def test_export_batches_progress(cache_dir, copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE_MOCKED_TEST_CASE)

    shutil.copytree(
        cache_dir / "mocked_multi_batch_output",
        "mocked_multi_batch_output",
        dirs_exist_ok=True,
    )

    # Add export section to config
    config.export = ExportConfig(discard_gradient=True, batches=[2])

    df = export_with_progress(config)
    # Check only simulations from given batches are present in export
    for id in df["batch"].values:
        assert id == 2


def test_export_nothing_for_empty_batch_list(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    os.makedirs(config.optimization_output_dir)
    shutil.copy(
        os.path.join(CASHED_RESULTS_FOLDER, "seba.db"),
        os.path.join(config.optimization_output_dir, "seba.db"),
    )

    # Add discard gradient flag to config file
    config.export = ExportConfig(
        discard_gradient=True, discard_rejected=True, batches=[]
    )
    df = export(config)

    # Check export returns empty data frame
    assert df.empty


def test_export_nothing(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    os.makedirs(config.optimization_output_dir)
    shutil.copy(
        os.path.join(CASHED_RESULTS_FOLDER, "seba.db"),
        os.path.join(config.optimization_output_dir, "seba.db"),
    )

    # Add discard gradient flag to config file
    config.export = ExportConfig(
        skip_export=True, discard_gradient=True, discard_rejected=True, batches=[3]
    )
    df = export(config)

    # Check export returns empty data frame
    assert df.empty


def test_get_export_path(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)

    # Test default export path when no csv_output_filepath is defined
    expected_export_path = os.path.join(
        config.output_dir, CONFIG_FILE.replace(".yml", ".csv")
    )
    assert expected_export_path == config.export_path

    # Test export path when csv_output_filepath is an absolute path
    new_export_folderpath = os.path.join(config.output_dir, "new/folder")
    new_export_filepath = os.path.join(
        new_export_folderpath, CONFIG_FILE.replace(".yml", ".csv")
    )

    config.export = ExportConfig(csv_output_filepath=new_export_filepath)

    expected_export_path = new_export_filepath
    assert expected_export_path == config.export_path

    # Test export path when csv_output_filepath is a relative path
    config.export.csv_output_filepath = os.path.join(
        "new/folder", CONFIG_FILE.replace(".yml", ".csv")
    )
    assert expected_export_path == config.export_path

    # Test export when file does not contain an extension.
    config_file_no_extension = os.path.splitext(os.path.basename(CONFIG_FILE))[0]
    shutil.copy(CONFIG_FILE, config_file_no_extension)
    new_config = EverestConfig.load_file(config_file_no_extension)
    expected_export_path = os.path.join(
        new_config.output_dir, f"{config_file_no_extension}.csv"
    )
    assert expected_export_path == new_config.export_path


@pytest.mark.fails_on_macos_github_workflow
def test_validate_export(cache_dir, copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE_MOCKED_TEST_CASE)

    shutil.copytree(
        cache_dir / "mocked_multi_batch_output",
        "mocked_multi_batch_output",
        dirs_exist_ok=True,
    )

    def check_error(expected_error, reported_errors):
        expected_error_msg, expected_export_ecl = expected_error
        error_list, export_ecl = reported_errors
        # If no error was message provided the list of errors
        # should also be empty
        if not expected_error_msg:
            assert len(error_list) == 0
            assert expected_export_ecl == export_ecl
        else:
            found = False
            for error in error_list:
                if expected_error_msg in error:
                    found = True
                    break
            assert found
            assert expected_export_ecl == export_ecl

    # Test export validator outputs no errors when the config file contains
    # an empty export section
    config.export = None
    check_error(("", True), validate_export(config))

    # Test error when user defines an empty list for the eclipse keywords
    config.export = ExportConfig()
    config.export.keywords = []
    check_error(
        ("No eclipse keywords selected for export", False), validate_export(config)
    )

    # Test error when user defines an empty list for the eclipse keywords
    # and empty list of for batches to export
    config.export.batches = []
    check_error(("No batches selected for export.", False), validate_export(config))

    # Test export validator outputs no errors when the config file contains
    # only keywords that represent a subset of already internalized keys
    config.export.keywords = ["FOPT"]
    config.export.batches = None
    check_error(("", True), validate_export(config))

    non_int_key = "STANGE_KEY"
    config.export.keywords = [non_int_key, "FOPT"]
    check_error(
        (
            "Non-internalized ecl keys selected for export '{keys}'." "".format(
                keys=non_int_key
            ),
            False,
        ),
        validate_export(config),
    )

    # Test that validating the export spots non-valid batches and removes
    # them from the list of batches selected for export.
    non_valid_batch = 42
    config.export = ExportConfig(batches=[0, non_valid_batch])
    check_error(
        (
            "Batch {} not found in optimization results. Skipping for"
            " current export".format(non_valid_batch),
            True,
        ),
        validate_export(config),
    )
    assert config.export.batches == [0]


def test_export_gradients(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    os.makedirs(config.optimization_output_dir)
    shutil.copy(
        os.path.join(CASHED_RESULTS_FOLDER, "seba.db"),
        os.path.join(config.optimization_output_dir, "seba.db"),
    )

    df = export(config)

    for function in config.objective_functions:
        for control in config.controls:
            for variable in control.variables:
                assert (
                    f"gradient-{function.name}-{control.name}_{variable.name}"
                    in df.columns
                )
