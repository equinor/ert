import os

import pytest

from ert._c_wrappers.enkf import EnkfFs


def test_create(copy_case):
    copy_case("local/mini_ert")
    mount_point = "fail_storage/ertensemble/default"

    assert os.path.exists(mount_point)
    fs = EnkfFs(mount_point)
    assert fs.refCount() == 1
    fs.umount()

    assert not os.path.exists("newFS")
    fs = EnkfFs.createFileSystem("newFS")
    assert os.path.exists("newFS")
    assert fs is None


def test_create2(tmpdir):
    with tmpdir.as_cwd():
        new_fs = EnkfFs.createFileSystem("newFS", mount=True)
        assert isinstance(new_fs, EnkfFs)


def test_throws():
    with pytest.raises(ValueError):
        EnkfFs("/does/not/exist")
