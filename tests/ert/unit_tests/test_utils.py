import os.path

from ert.utils import makedirs_if_needed


def test_makedirs(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # assert output dir (/tmp/tmpXXXX) is empty
    assert not os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 0

    # create output folder
    makedirs_if_needed(output_dir)

    # assert output folder created
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1


def test_makedirs_already_exists(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # create outputfolder and verify it's existing
    makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1

    # run makedirs_if_needed again, verify nothing happened
    makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1


def test_makedirs_roll_existing(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # create outputfolder and verify it's existing
    makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1

    # run makedirs_if_needed again, verify old dir rolled
    makedirs_if_needed(output_dir, True)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 2

    # run makedirs_if_needed again, verify old dir rolled
    makedirs_if_needed(output_dir, True)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 3
