from everest.config.everest_config import EverestConfig, get_system_installed_jobs
from everest.jobs import shell_commands

from tests.everest.utils import relpath


def test_everest_shell_commands_list():
    # Check list of defined shell commands are part of the list of ert
    # installed system jobs
    system_installed_jobs = get_system_installed_jobs()
    for command_name in shell_commands:
        assert command_name in system_installed_jobs


def test_default_shell_scripts():
    config_path = relpath("test_data", "shell_commands", "config_shell_commands.yml")
    # Check that a config file containing default shell jobs as part of the
    # forward-model section is valid
    EverestConfig.load_file(config_path)
