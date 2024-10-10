import os
import os.path

from .run_cli import run_cli_with_pm


def test_shell_scripts_integration(tmpdir):
    """
    The following test is a regression test that
    checks that the scripts under src/ert/resources/shell_scripts
    are not broken, and correctly installed through site-config.
    """
    with tmpdir.as_cwd():
        ert_config_fname = "test.ert"
        with open(ert_config_fname, "w", encoding="utf-8") as file_h:
            file_h.write(
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
"""
            )

        with open("file.txt", "w", encoding="utf-8") as file_h:
            file_h.write("something")

        run_cli_with_pm(["test_run", "--disable-monitor", ert_config_fname])

        with open("realization-0/iter-0/moved.txt", encoding="utf-8") as output_file:
            assert output_file.read() == "something"
        assert not os.path.exists("realization-0/iter-0/copied.txt")
        assert not os.path.exists("realization-0/iter-0/copied2.txt")
        assert os.path.exists("realization-0/iter-0/copied3.txt")
        assert not os.path.exists("realization-0/iter-0/mydir")
        assert os.path.exists("realization-0/iter-0/mydir2")
        assert not os.path.exists("realization-0/iter-0/mydir3")
        assert os.path.exists("realization-0/iter-0/mydir4/mydir3")
        assert os.path.exists("realization-0/iter-0/mydir4/mydir3/copied.txt")
