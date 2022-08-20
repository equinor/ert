import os

from ert._c_wrappers.enkf import EnkfFs, ResConfig


def test_create(copy_case):
    copy_case("mini_ert")
    mount_point = "fail_storage/ertensemble/default"
    res_config = ResConfig("mini_fail_config")
    assert os.path.exists(mount_point)
    fs = EnkfFs(mount_point, res_config.ensemble_config, True, 10)

    assert not os.path.exists("newFS")
    fs = EnkfFs.createFileSystem("newFS", res_config.ensemble_config, True, 10)
    assert os.path.exists("newFS")
    assert isinstance(fs, EnkfFs)
