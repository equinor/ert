import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from everest import filter_data
from everest.bin.utils import export_with_progress
from everest.config import EverestConfig
from everest.config.export_config import ExportConfig
from everest.export import check_for_errors, export_data

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


def test_export_only_non_gradient_with_increased_merit(cached_example, snapshot):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)
    # Default export functionality when no export section is defined
    df = export_data(
        export_config=config.export,
        output_dir=config.output_dir,
        data_file=config.model.data_file if config.model else None,
    )

    # Test that the default export functionality generated data frame
    # contains only non gradient simulations
    snapshot.assert_match(
        df.drop(["start_time", "end_time"], axis=1).round(4).to_csv(), "export.csv"
    )


def test_export_only_non_gradient(cached_example, snapshot):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    # Add export section to config
    config.export = ExportConfig(discard_rejected=False)

    df = export_data(
        export_config=config.export,
        output_dir=config.output_dir,
        data_file=config.model.data_file if config.model else None,
    )

    snapshot.assert_match(
        df.drop(["start_time", "end_time"], axis=1).round(4).to_csv(), "export.csv"
    )


def test_export_only_increased_merit(cached_example, snapshot):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    # Add export section to config
    config.export = ExportConfig(discard_gradient=False)

    df = export_data(
        export_config=config.export,
        output_dir=config.output_dir,
        data_file=config.model.data_file if config.model else None,
    )

    snapshot.assert_match(
        df.drop(["start_time", "end_time"], axis=1).round(4).to_csv(),
        "export.csv",
    )


def test_export_all_batches(cached_example, snapshot):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    # Add export section to config
    config.export = ExportConfig(discard_gradient=False, discard_rejected=False)

    df = export_data(
        export_config=config.export,
        output_dir=config.output_dir,
        data_file=config.model.data_file if config.model else None,
    )

    snapshot.assert_match(
        df.drop(["start_time", "end_time"], axis=1).round(4).to_csv(), "export.csv"
    )


def test_export_only_give_batches(cached_example, snapshot):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    # Add export section to config
    config.export = ExportConfig(discard_gradient=True, batches=[2])

    df = export_data(
        export_config=config.export,
        output_dir=config.output_dir,
        data_file=config.model.data_file if config.model else None,
    )

    snapshot.assert_match(
        df.drop(["start_time", "end_time"], axis=1).round(4).to_csv(), "export.csv"
    )


def test_export_batches_progress(cached_example, snapshot):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    # Add export section to config
    config.export = ExportConfig(discard_gradient=True, batches=[2])

    df = export_with_progress(config)
    # Check only simulations from given batches are present in export
    # drop non-deterministic columns
    df = df.drop(["start_time", "end_time", "simulation"], axis=1)
    df = df.sort_values(by=["realization", "batch", "sim_avg_obj"])

    snapshot.assert_match(df.round(4).to_csv(index=False), "export.csv")


def test_export_nothing_for_empty_batch_list(cached_example):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    # Add discard gradient flag to config file
    config.export = ExportConfig(
        discard_gradient=True, discard_rejected=True, batches=[]
    )
    df = export_data(
        export_config=config.export,
        output_dir=config.output_dir,
        data_file=config.model.data_file if config.model else None,
    )

    # Check export returns empty data frame
    assert df.empty


def test_export_nothing(cached_example):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    # Add discard gradient flag to config file
    config.export = ExportConfig(
        skip_export=True, discard_gradient=True, discard_rejected=True, batches=[3]
    )
    df = export_data(
        export_config=config.export,
        output_dir=config.output_dir,
        data_file=config.model.data_file if config.model else None,
    )

    # Check export returns empty data frame
    assert df.empty


def test_get_export_path(cached_example):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

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


def test_validate_export(cached_example):
    config_path, config_file, _ = cached_example(
        "../../tests/everest/test_data/mocked_test_case/mocked_multi_batch.yml"
    )
    config = EverestConfig.load_file(Path(config_path) / config_file)

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

    # Test error when user defines an empty list for the eclipse keywords
    config.export = ExportConfig()
    config.export.keywords = []
    errors, export_ecl = check_for_errors(
        config=config.export,
        optimization_output_path=config.optimization_output_dir,
        storage_path=config.storage_dir,
        data_file_path=config.model.data_file,
    )
    check_error(
        expected_error=("No eclipse keywords selected for export", False),
        reported_errors=(errors, export_ecl),
    )

    # Test error when user defines an empty list for the eclipse keywords
    # and empty list of for batches to export
    config.export.batches = []
    errors, export_ecl = check_for_errors(
        config=config.export,
        optimization_output_path=config.optimization_output_dir,
        storage_path=config.storage_dir,
        data_file_path=config.model.data_file,
    )
    check_error(
        expected_error=("No batches selected for export.", False),
        reported_errors=(errors, export_ecl),
    )

    # Test export validator outputs no errors when the config file contains
    # only keywords that represent a subset of already internalized keys
    config.export.keywords = ["FOPT"]
    config.export.batches = None
    errors, export_ecl = check_for_errors(
        config=config.export,
        optimization_output_path=config.optimization_output_dir,
        storage_path=config.storage_dir,
        data_file_path=config.model.data_file,
    )
    check_error(expected_error=("", True), reported_errors=(errors, export_ecl))

    non_int_key = "STANGE_KEY"
    config.export.keywords = [non_int_key, "FOPT"]
    errors, export_ecl = check_for_errors(
        config=config.export,
        optimization_output_path=config.optimization_output_dir,
        storage_path=config.storage_dir,
        data_file_path=config.model.data_file,
    )

    check_error(
        (
            "Non-internalized ecl keys selected for export '{keys}'." "".format(
                keys=non_int_key
            ),
            False,
        ),
        (errors, export_ecl),
    )

    # Test that validating the export spots non-valid batches and removes
    # them from the list of batches selected for export.
    non_valid_batch = 42
    config.export = ExportConfig(batches=[0, non_valid_batch])
    errors, export_ecl = check_for_errors(
        config=config.export,
        optimization_output_path=config.optimization_output_dir,
        storage_path=config.storage_dir,
        data_file_path=config.model.data_file,
    )
    check_error(
        (
            "Batch {} not found in optimization results. Skipping for"
            " current export".format(non_valid_batch),
            True,
        ),
        (errors, export_ecl),
    )
    assert config.export.batches == [0]


def test_export_gradients(cached_example, snapshot):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    df = export_data(
        export_config=config.export,
        output_dir=config.output_dir,
        data_file=config.model.data_file if config.model else None,
    )

    snapshot.assert_match(
        df.drop(["start_time", "end_time"], axis=1).round(4).to_csv(), "export.csv"
    )
