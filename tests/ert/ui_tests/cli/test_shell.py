from pathlib import Path

from .run_cli import run_cli


def test_shell_scripts_integration(tmpdir):
    """
    The following test is a regression test that
    checks that the scripts under src/ert/resources/shell_scripts
    are not broken, and correctly installed through site-config.
    """
    with tmpdir.as_cwd():
        ert_config_fname = "test.ert"
        Path(ert_config_fname).write_text(
            """
RUNPATH realization-<IENS>/iter-<ITER>
JOBNAME TEST
QUEUE_SYSTEM LOCAL
NUM_REALIZATIONS 1
FORWARD_MODEL COPY_FILE(<FROM>=<CONFIG_PATH>/file.txt, <TO>=copied.txt)
FORWARD_MODEL COPY_FILE(<FROM>=<CONFIG_PATH>/file.txt, <TO>=copied2.txt)
FORWARD_MODEL CAREFUL_COPY_FILE(<FROM>=<CONFIG_PATH>/file.txt, <TO>=copied3.txt)
FORWARD_MODEL MOVE_FILE(<FROM>=copied.txt, <TO>=moved.txt)
FORWARD_MODEL DELETE_FILE(<FILES>=copied2.txt)
FORWARD_MODEL MAKE_DIRECTORY(<DIRECTORY>=mydir)
FORWARD_MODEL COPY_DIRECTORY(<FROM>=mydir, <TO>=mydir2)
FORWARD_MODEL DELETE_DIRECTORY(<DIRECTORY>=mydir)
FORWARD_MODEL COPY_FILE(<FROM>=<CONFIG_PATH>/file.txt, <TO>=mydir3/copied.txt)
FORWARD_MODEL MOVE_DIRECTORY(<FROM>=mydir3, <TO>=mydir4/mydir3)
""",
            encoding="utf-8",
        )

        Path("file.txt").write_text("something", encoding="utf-8")

        run_cli("test_run", "--disable-monitoring", ert_config_fname)

        assert (
            Path("realization-0/iter-0/moved.txt").read_text(encoding="utf-8")
            == "something"
        )
        assert not Path("realization-0/iter-0/copied.txt").exists()
        assert not Path("realization-0/iter-0/copied2.txt").exists()
        assert Path("realization-0/iter-0/copied3.txt").exists()
        assert not Path("realization-0/iter-0/mydir").exists()
        assert Path("realization-0/iter-0/mydir2").exists()
        assert not Path("realization-0/iter-0/mydir3").exists()
        assert Path("realization-0/iter-0/mydir4/mydir3").exists()
        assert Path("realization-0/iter-0/mydir4/mydir3/copied.txt").exists()
