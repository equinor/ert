from ert._c_wrappers.enkf import EnKFMain, ResConfig


def mount_and_umount(ert, case_name):
    fs_manager = ert.storage_manager
    mount_point = fs_manager[case_name]
    del mount_point


def test_mount_fs(benchmark, template_config):
    with template_config["folder"].as_cwd():
        config = ResConfig("poly.ert")
        ert = EnKFMain(config)
        benchmark(mount_and_umount, ert, "default")
