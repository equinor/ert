from pathlib import Path
from unittest.mock import MagicMock

from ert.observation_converters import convert_observations
from ert.plugins import get_site_plugins


def test_that_convert_observations_does_not_fail_when_config_has_hooked_workflows(
    use_tmpdir,
):
    """This reproduces the case where ErtConfig.from_file() is called without
    plugins while hooked workflows reference plugin-provided jobs.
    """
    site_plugins = get_site_plugins()

    arbitrary_existing_job = next(iter(site_plugins.installed_workflow_jobs))

    workflow_file = Path("my_hook_workflow")
    workflow_file.write_text(f"{arbitrary_existing_job}\n", encoding="utf-8")

    obs_config = "foo.txt"
    summary_obs = (
        "SUMMARY_OBSERVATION { KEY = FOPR; VALUE = 10; ERROR = 5; DATE = 2000-01-01; };"
    )
    Path(obs_config).write_text(
        summary_obs,
        encoding="utf-8",
    )

    ert_config = "config.ert"
    minimal_workflow_config = f"""\
    NUM_REALIZATIONS 10
    ECLBASE foo
    OBS_CONFIG {obs_config}
    LOAD_WORKFLOW {workflow_file} MY_HOOK
    HOOK_WORKFLOW MY_HOOK PRE_SIMULATION
    """
    Path(ert_config).write_text(
        minimal_workflow_config,
        encoding="utf-8",
    )

    for format_ in ["summary", "bulk", "yaml"]:
        args = MagicMock(format=format_, config=ert_config)
        convert_observations(args, site_plugins)
