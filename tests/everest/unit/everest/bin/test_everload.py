import os
import shutil
from unittest.mock import patch

import pandas as pd
import pytest
from tests.everest.utils import (
    capture_streams,
    create_cached_mocked_test_case,
)

from ert.config import ErtConfig
from everest import MetaDataColumnNames as MDCN
from everest import export
from everest.bin.everload_script import everload_entry
from everest.config import EverestConfig
from everest.strings import STORAGE_DIR

CONFIG_FILE = "mocked_multi_batch.yml"

pytestmark = pytest.mark.xdist_group(name="starts_everest")


@pytest.fixture
def cache_dir(request, monkeypatch):
    return create_cached_mocked_test_case(request, monkeypatch)


def get_config(cache_dir):
    shutil.copytree(
        cache_dir / "mocked_multi_batch_output", "mocked_multi_batch_output"
    )
    config = EverestConfig.load_file(CONFIG_FILE)
    simdir = config.simulation_dir

    # Assume there is already a storage
    assert os.path.isdir(config.storage_dir)

    # Create the simulation folder
    if not os.path.isdir(simdir):
        os.makedirs(simdir)

    return config


def assertInternalizeCalls(batch_ids, mocked_internalize):
    for i, b_id in enumerate(batch_ids):
        config, bid, data = mocked_internalize.call_args_list[i].args
        assert isinstance(config, ErtConfig)
        assert isinstance(data, pd.DataFrame)
        assert bid == b_id


def assertBackup(config: EverestConfig):
    backupdir = [
        d for d in os.listdir(config.output_dir) if d.startswith(STORAGE_DIR + "__")
    ]
    assert backupdir != []


@patch("everest.bin.everload_script._internalize_batch")
@pytest.mark.fails_on_macos_github_workflow
def test_everload_entry_run(
    mocked_internalize, cache_dir, copy_mocked_test_data_to_tmp
):
    """Test running everload on an optimization case"""
    config = get_config(cache_dir)
    everload_entry([CONFIG_FILE, "-s"])

    df = export(config, export_ecl=False)
    batch_ids = set(df[MDCN.BATCH])
    assertInternalizeCalls(batch_ids, mocked_internalize)
    assertBackup(config)


@patch("everest.bin.everload_script._internalize_batch")
@pytest.mark.fails_on_macos_github_workflow
def test_everload_entry_run_empty_batch_list(_, copy_mocked_test_data_to_tmp):
    """Test running everload on an optimization case"""
    with pytest.raises(SystemExit), capture_streams() as (_, err):
        everload_entry([CONFIG_FILE, "-s", "-b"])
        assert (
            "error: argument -b/--batches: expected at least one argument"
            in err.getvalue()
        )


@patch("everest.bin.everload_script._internalize_batch")
@pytest.mark.fails_on_macos_github_workflow
def test_everload_entry_missing_folders(
    mocked_internalize, cache_dir, copy_mocked_test_data_to_tmp
):
    """Test running everload when output folders are missing"""
    config = get_config(cache_dir)
    shutil.rmtree(config.simulation_dir)
    with pytest.raises(RuntimeError, match="simulation"):
        everload_entry([CONFIG_FILE, "-s"])
    shutil.rmtree(config.output_dir)
    with pytest.raises(RuntimeError, match="never run"):
        everload_entry([CONFIG_FILE, "-s"])
    mocked_internalize.assert_not_called()


@patch("everest.bin.everload_script._internalize_batch")
@pytest.mark.fails_on_macos_github_workflow
def test_everload_entry_batches(
    mocked_internalize, cache_dir, copy_mocked_test_data_to_tmp
):
    """Test running everload with a selection of batches"""
    config = get_config(cache_dir)
    # pick every second batch (assume there are at least 2)
    df = export(config, export_ecl=False)
    batch_ids = list(set(df[MDCN.BATCH]))
    assert len(batch_ids) > 1
    batch_ids = batch_ids[::2]

    everload_entry([CONFIG_FILE, "-s", "-b"] + [str(b) for b in batch_ids])

    assertInternalizeCalls(batch_ids, mocked_internalize)
    assertBackup(config)


@patch("everest.bin.everload_script._internalize_batch")
@pytest.mark.fails_on_macos_github_workflow
def test_everload_entry_invalid_batches(
    mocked_internalize, copy_mocked_test_data_to_tmp
):
    """Test running everload with no or wrong batches"""
    with pytest.raises(SystemExit), capture_streams() as (_, err):
        everload_entry([CONFIG_FILE, "-s", "-b", "-2", "5412"])
        assert "error: Invalid batch given: '-2'" in err.getvalue()

    with pytest.raises(SystemExit), capture_streams() as (_, err):
        everload_entry([CONFIG_FILE, "-s", "-b", "0123"])
        assert "error: Invalid batch given: '0123'" in err.getvalue()

    mocked_internalize.assert_not_called()


@patch("everest.bin.everload_script._internalize_batch")
@pytest.mark.fails_on_macos_github_workflow
def test_everload_entry_overwrite(
    mocked_internalize, cache_dir, copy_mocked_test_data_to_tmp
):
    """Test running everload with the --overwrite flag"""
    config = get_config(cache_dir)
    everload_entry([CONFIG_FILE, "-s", "--overwrite"])

    df = export(config, export_ecl=False)
    batch_ids = set(df[MDCN.BATCH])
    assertInternalizeCalls(batch_ids, mocked_internalize)

    # Note that, as we are mocking the entire ert related part, the
    # internalization does not take place, so no new storage dir is created
    backupdir = [d for d in os.listdir(config.output_dir) if d.startswith(STORAGE_DIR)]
    assert backupdir == []


@patch("everest.bin.everload_script._internalize_batch")
@pytest.mark.fails_on_macos_github_workflow
def test_everload_entry_not_silent(
    mocked_internalize, cache_dir, copy_mocked_test_data_to_tmp
):
    """Test running everload without the -s flag"""
    config = get_config(cache_dir)

    no = lambda _: "n"  # pylint:disable=unnecessary-lambda-assignment
    yes = lambda _: "y"  # pylint:disable=unnecessary-lambda-assignment

    with capture_streams() as (stdout, _):
        with patch("everest.bin.everload_script.input", side_effect=no):
            everload_entry([CONFIG_FILE])
        assert "backed up" in stdout.getvalue()
    mocked_internalize.assert_not_called()

    with capture_streams() as (stdout, _):
        with patch("everest.bin.everload_script.input", side_effect=no):
            everload_entry([CONFIG_FILE, "--overwrite"])
        assert "WARNING" in stdout.getvalue()
    mocked_internalize.assert_not_called()

    with capture_streams() as (stdout, _):
        with patch("everest.bin.everload_script.input", side_effect=yes):
            everload_entry([CONFIG_FILE])
        assert len(stdout.getvalue()) > 0
    df = export(config, export_ecl=False)
    batch_ids = set(df[MDCN.BATCH])
    assertInternalizeCalls(batch_ids, mocked_internalize)

    df = export(config, export_ecl=False)
    batch_ids = set(df[MDCN.BATCH])
    assertInternalizeCalls(batch_ids, mocked_internalize)
    assertBackup(config)
