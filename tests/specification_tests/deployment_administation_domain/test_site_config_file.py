import os
import os.path
import stat
from pathlib import Path

from ert._c_wrappers.enkf import ResConfig


def test_that_job_script_can_be_set_in_site_config(monkeypatch, tmp_path):
    """
    We use the jobscript field to inject a komodo environment onprem.
    This overwrites the value by appending to the default siteconfig.
    Need to check that the second JOB_SCRIPT is the one that gets used.
    """

    # As an ert system administator

    # WHEN I set the JOB_SCRIPT in a site config
    test_site_config = tmp_path / "test_site_config.ert"
    my_script = (tmp_path / "my_script").resolve()
    my_script.write_text("")
    st = os.stat(my_script)
    os.chmod(my_script, st.st_mode | stat.S_IEXEC)
    test_site_config.write_text(
        f"JOB_SCRIPT job_dispatch.py\nJOB_SCRIPT {my_script}\nQUEUE_SYSTEM LOCAL\n"
    )

    # AND Specify the site config file with an environment variable
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))

    # THEN I expect that the job script to be set in all ert configurations
    test_user_config = tmp_path / "user_config.ert"
    test_user_config.write_text(
        "JOBNAME  Job%d\nRUNPATH /tmp/simulations/run%d\nNUM_REALIZATIONS 10\n"
    )
    res_config = ResConfig(str(test_user_config))
    assert Path(res_config.queue_config.job_script).resolve() == my_script
