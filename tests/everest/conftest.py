import os
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest
import yaml

from ert.config import ConfigWarning, QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.config.control_config import ControlConfig
from everest.detached.jobs import everserver
from tests.everest.utils import relpath


@pytest.fixture(autouse=True)
def filter_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*Forward model might not write the required output.*",
        category=ConfigWarning,
    )


@pytest.fixture(scope="session")
def testdata() -> Path:
    return Path(__file__).parent / "test_data"


@pytest.fixture
def copy_testdata_tmpdir(
    testdata: Path, tmp_path: Path
) -> Iterator[Callable[[str | None], Path]]:
    def _copy_tree(path: str | None = None):
        path_ = testdata if path is None else testdata / path
        shutil.copytree(path_, tmp_path, dirs_exist_ok=True)
        return path_

    cwd = Path.cwd()
    os.chdir(tmp_path)
    yield _copy_tree
    os.chdir(cwd)


@pytest.fixture(scope="module")
def control_data_no_variables() -> dict[str, str | float]:
    return {
        "name": "group_0",
        "type": "well_control",
        "min": 0.0,
        "max": 0.1,
        "perturbation_magnitude": 0.005,
    }


@pytest.fixture(
    scope="module",
    params=(
        pytest.param(
            [
                {"name": "w00", "initial_guess": 0.0626, "index": 0},
                {"name": "w00", "initial_guess": 0.063, "index": 1},
                {"name": "w00", "initial_guess": 0.0617, "index": 2},
                {"name": "w00", "initial_guess": 0.0621, "index": 3},
                {"name": "w01", "initial_guess": 0.0627, "index": 0},
                {"name": "w01", "initial_guess": 0.0631, "index": 1},
                {"name": "w01", "initial_guess": 0.0618, "index": 2},
                {"name": "w01", "initial_guess": 0.0622, "index": 3},
                {"name": "w02", "initial_guess": 0.0628, "index": 0},
                {"name": "w02", "initial_guess": 0.0632, "index": 1},
                {"name": "w02", "initial_guess": 0.0619, "index": 2},
                {"name": "w02", "initial_guess": 0.0623, "index": 3},
                {"name": "w03", "initial_guess": 0.0629, "index": 0},
                {"name": "w03", "initial_guess": 0.0633, "index": 1},
                {"name": "w03", "initial_guess": 0.062, "index": 2},
                {"name": "w03", "initial_guess": 0.0624, "index": 3},
            ],
            id="indexed variables",
        ),
        pytest.param(
            [
                {"name": "w00", "initial_guess": [0.0626, 0.063, 0.0617, 0.0621]},
                {"name": "w01", "initial_guess": [0.0627, 0.0631, 0.0618, 0.0622]},
                {"name": "w02", "initial_guess": [0.0628, 0.0632, 0.0619, 0.0623]},
                {"name": "w03", "initial_guess": [0.0629, 0.0633, 0.062, 0.0624]},
            ],
            id="vectored variables",
        ),
    ),
)
def control_config(
    request,
    control_data_no_variables: dict[str, str | float],
) -> ControlConfig:
    config = deepcopy(control_data_no_variables)
    config["variables"] = request.param
    return ControlConfig.model_validate(config)


@pytest.fixture
def copy_math_func_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("..", "..", "test-data", "everest", "math_func")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_mocked_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("test_data", "mocked_test_case")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("test_data")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_template_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("test_data", "templating")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_egg_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("..", "..", "test-data", "everest", "egg")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def change_to_tmpdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def evaluator_server_config_generator():
    def create_evaluator_server_config(run_model):
        return EvaluatorServerConfig(
            custom_port_range=range(49152, 51819)
            if run_model.ert_config.queue_config.queue_system == QueueSystem.LOCAL
            else None
        )

    return create_evaluator_server_config


@pytest.fixture
def cached_example(pytestconfig, evaluator_server_config_generator):
    cache = pytestconfig.cache

    def run_config(test_data_case: str):
        if cache.get(f"cached_example:{test_data_case}", None) is None:
            my_tmpdir = Path(tempfile.mkdtemp())
            config_path = (
                Path(__file__) / f"../../../test-data/everest/{test_data_case}"
            ).resolve()
            config_file = config_path.name

            shutil.copytree(config_path.parent, my_tmpdir / "everest")
            config = EverestConfig.load_file(my_tmpdir / "everest" / config_file)
            run_model = EverestRunModel.create(config)
            evaluator_server_config = evaluator_server_config_generator(run_model)
            try:
                run_model.run_experiment(evaluator_server_config)
            except Exception as e:
                raise Exception(f"Failed running {config_path} with error: {e}") from e

            result_path = my_tmpdir / "everest"

            optimal_result = run_model.result
            optimal_result_json = {
                "batch": optimal_result.batch,
                "controls": optimal_result.controls,
                "total_objective": optimal_result.total_objective,
            }

            cache.set(
                f"cached_example:{test_data_case}",
                (str(result_path), config_file, optimal_result_json),
            )

        result_path, config_file, optimal_result_json = cache.get(
            f"cached_example:{test_data_case}", (None, None, None)
        )

        copied_tmpdir = tempfile.mkdtemp()
        shutil.copytree(result_path, Path(copied_tmpdir) / "everest")
        copied_path = str(Path(copied_tmpdir) / "everest")
        os.chdir(copied_path)

        return copied_path, config_file, optimal_result_json

    return run_config


@pytest.fixture
def min_config():
    yield yaml.safe_load(
        dedent("""
    model: {"realizations": [0]}
    controls:
      -
        name: my_control
        type: well_control
        min: 0
        max: 0.1
        variables:
          - { name: test, initial_guess: 0.1 }
    objective_functions:
      - {name: my_objective}
    config_path: .
    """)
    )


@pytest.fixture()
def mock_server(monkeypatch):
    monkeypatch.setattr(everserver, "_configure_loggers", MagicMock())
    monkeypatch.setattr(everserver, "_generate_authentication", MagicMock())
    monkeypatch.setattr(
        everserver, "_generate_certificate", lambda *args: (None, None, None)
    )
    monkeypatch.setattr(everserver, "_find_open_port", lambda *args, **kwargs: 42)
    monkeypatch.setattr(everserver, "_write_hostfile", MagicMock())
    monkeypatch.setattr(everserver, "_everserver_thread", MagicMock())
    monkeypatch.setattr(everserver, "export_to_csv", MagicMock())
    monkeypatch.setattr(everserver, "export_with_progress", MagicMock())
