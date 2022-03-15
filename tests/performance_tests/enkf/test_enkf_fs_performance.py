from res.enkf import EnKFMain


def mount_and_umount(ert, case_name):
    fs_manager = ert.getEnkfFsManager()
    assert not fs_manager.isCaseMounted(case_name)
    mount_point = fs_manager.getFileSystem(case_name)
    assert fs_manager.isCaseMounted(case_name)
    del mount_point
    fs_manager.umount()


def test_mount_fs(setup_case, benchmark):
    res_config = setup_case("local/snake_oil", "snake_oil.ert")

    ert = EnKFMain(res_config)
    benchmark(mount_and_umount, ert, "default_1")
