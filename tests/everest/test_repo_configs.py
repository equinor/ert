import os

import pytest
from ropt.workflow import find_optimizer_plugin

from everest.config_file_loader import yaml_file_to_substituted_config_dict


def _get_all_files(folder):
    return [
        os.path.join(root, filename)
        for root, _, files in os.walk(folder)
        for filename in files
    ]


@pytest.mark.slow
def test_all_repo_configs():
    repo_dir = os.path.join(os.path.dirname(__file__), "..")
    repo_dir = os.path.realpath(repo_dir)

    data_folders = (
        "examples/eightcells/everest/input",
        "everest/data",
    )

    config_folders = (
        "examples",
        "everest",
        "tests",
    )
    config_folders = map(lambda fn: os.path.join(repo_dir, fn), config_folders)  # noqa E731

    def is_yaml(fn):
        return fn.endswith(".yml")

    def is_data(fn):
        return any(df in fn for df in data_folders)

    def is_config(fn):
        return is_yaml(fn) and not is_data(fn) and "invalid" not in fn

    config_files = [fn for cdir in config_folders for fn in _get_all_files(cdir)]
    config_files = filter(is_config, config_files)

    if find_optimizer_plugin("scipy/default") is None:
        config_files = [f for f in config_files if "scipy" not in f]

    config_files = list(config_files)
    for config_file in config_files:
        config = yaml_file_to_substituted_config_dict(config_file)
        assert config is not None
