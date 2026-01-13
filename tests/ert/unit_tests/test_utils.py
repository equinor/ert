from pathlib import Path

from ert.utils import makedirs_if_needed


def test_makedirs(change_to_tmpdir):
    output_dir = Path("unittest_everest_output")
    cwd = Path.cwd()

    # assert output dir (/tmp/tmpXXXX) is empty
    assert not output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 0

    # create output folder
    makedirs_if_needed(output_dir)

    # assert output folder created
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 1


def test_makedirs_already_exists(change_to_tmpdir):
    output_dir = Path("unittest_everest_output")
    cwd = Path.cwd()

    # create outputfolder and verify it's existing
    makedirs_if_needed(output_dir)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 1

    # run makedirs_if_needed again, verify nothing happened
    makedirs_if_needed(output_dir)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 1


def test_makedirs_roll_existing(change_to_tmpdir):
    output_dir = Path("unittest_everest_output")
    cwd = Path.cwd()

    # create outputfolder and verify it's existing
    makedirs_if_needed(output_dir)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 1

    # run makedirs_if_needed again, verify old dir rolled
    makedirs_if_needed(output_dir, True)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 2

    # run makedirs_if_needed again, verify old dir rolled
    makedirs_if_needed(output_dir, True)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 3
