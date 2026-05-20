from pathlib import Path

import pytest
from ropt.workflow import find_backend_plugin

from everest.config_file_loader import yaml_file_to_substituted_config_dict


def _get_all_files(folder: Path) -> list[Path]:
    return [path for path in folder.rglob("*") if path.is_file()]


@pytest.mark.slow
def test_all_repo_configs():
    repo_dir = Path(__file__).parent.parent.resolve()

    data_folders = (
        "examples/eightcells/everest/input",
        "everest/data",
    )

    config_folders = (
        repo_dir / "examples",
        repo_dir / "everest",
        repo_dir / "tests",
    )

    def is_yaml(fn: str):
        return fn.endswith(".yml")

    def is_data(fn: str):
        return any(df in fn for df in data_folders)

    def is_config(fn: str):
        return is_yaml(fn) and not is_data(fn) and "invalid" not in fn

    config_files: list[str] = [
        str(fn) for cdir in config_folders for fn in _get_all_files(cdir)
    ]
    config_files = filter(is_config, config_files)

    if find_backend_plugin("scipy/default") is None:
        config_files = [f for f in config_files if "scipy" not in str(f)]

    config_files = list(config_files)
    for config_file in config_files:
        config = yaml_file_to_substituted_config_dict(str(config_file))
        assert config is not None
