import json
import os
from pathlib import Path

import pytest
from ruamel.yaml import YAML

import everest
from ert.base_model_context import use_runtime_plugins
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import get_site_plugins
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig, InstallTemplateConfig
from tests.ert.unit_tests.resources._import_from_location import import_from_location
from tests.ert.utils import SOURCE_DIR
from tests.everest.utils import everest_config_with_defaults

CONFIG = {
    "controls": [
        {
            "name": "well_drill",
            "min": 0,
            "max": 1,
            "perturbation_magnitude": 0.01,
            "variables": [
                {"name": "PROD1", "initial_guess": 1},
                {"name": "PROD2", "initial_guess": 0.9},
            ],
        }
    ],
    "objective_functions": [{"name": "objectf_by_tmpl"}],
    "optimization": {"algorithm": "optpp_q_newton", "max_iterations": 1},
    "model": {"realizations": [0]},
    "install_templates": [
        {
            "template": "<CONFIG_PATH>/well_drill_info.tmpl",
            "output_file": "well_drill.json",
        },
        {
            "template": "<CONFIG_PATH>/the_optimal_template.tmpl",
            "output_file": "objectf_by_tmpl",
        },
    ],
}
WELL_DRILL_TMPL = """PROD1 takes value {{ well_drill.PROD1 }}, implying \
{{ "on" if well_drill.PROD1 >= 0.5 else "off" }}
PROD2 takes value {{ well_drill.PROD2 }}, implying \
{{ "on" if well_drill.PROD2 >= 0.5 else "off" }}
----------------------------------
{%- for well_name, value in well_drill.items() %}
{{ well_name }} takes value {{ value }}, implying {{ "on" if value >= 0.5 else "off"}}
{%- endfor %}
"""
THE_OPTIMAL_TEMPLATE_TMPL = "{{ well_drill.values() | sum() }}"


template_render = import_from_location(
    "template_render",
    os.path.join(
        SOURCE_DIR,
        "src/ert/resources/forward_models/template_render.py",
    ),
)


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_install_template(change_to_tmpdir):
    YAML(typ="safe", pure=True).dump(CONFIG, Path("config.yml"))
    Path("well_drill_info.tmpl").write_text(WELL_DRILL_TMPL, encoding="utf-8")
    Path("the_optimal_template.tmpl").write_text(
        THE_OPTIMAL_TEMPLATE_TMPL, encoding="utf-8"
    )
    config = EverestConfig.load_file("config.yml")
    site_plugins = get_site_plugins()
    with use_runtime_plugins(get_site_plugins()):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)


def test_well_order_template(change_to_tmpdir):
    order_tmpl = everest.templates.fetch_template("well_order.tmpl")

    well_order = {
        "PROD1": 0.3,
        "PROD2": 0.1,
        "PROD3": 0.2,
        "INJECT1": 0.5,
        "INJECT2": 0.01,
        "SUPER_WELL1": 1,
        "YET_ANOTHER_WELL": 1,
    }

    data_file = "well_order.json"
    Path(data_file).write_text(json.dumps(well_order), encoding="utf-8")

    output_file = "well_order_list.json"
    template_render.render_template(
        data_file,
        order_tmpl,
        output_file,
    )

    order = json.loads(Path(output_file).read_text(encoding="utf-8"))

    assert len(well_order) == len(order)
    for idx in range(len(order) - 1):
        assert well_order[order[idx]] <= well_order[order[idx + 1]]


@pytest.mark.integration_test
@pytest.mark.parametrize("test", ["install_templates", "template_render"])
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_user_specified_data_n_template(copy_math_func_test_data_to_tmp, test):
    """
    Ensure that a user specifying a data resource and an installed_template
    with "extra_data", the results of that template will be passed to the
    directory for each consecutive simulation
    """

    config = EverestConfig.load_file("config_minimal.yml")

    # Write out some constants to a yaml file; doing it here, so config
    # test (TestRepoConfigs) doesn't try to lint this yaml file.
    YAML(typ="safe", pure=True).dump(
        {"CONST1": "VALUE1", "CONST2": "VALUE2"}, Path("my_constants.yml")
    )

    # Write out the template to which takes the constants above
    Path("my_constants.tmpl").write_text(
        "{{ my_constants.CONST1 }}+{{ my_constants.CONST2 }}", encoding="utf-8"
    )

    # Modify the minimal config with template and constants
    updated_config_dict = config.to_dict()
    updated_config_dict.update(
        {
            "optimization": {
                "algorithm": "optpp_q_newton",
                "convergence_tolerance": 0.005,
                "max_iterations": 1,
                "perturbation_num": 1,
                "max_function_evaluations": 1,
            },
            "install_data": [
                {
                    "source": "<CONFIG_PATH>/my_constants.yml",
                    "target": "my_constants.yml",
                }
            ],
        }
    )
    match test:
        case "install_templates":
            updated_config_dict["install_templates"] = [
                {
                    "template": "<CONFIG_PATH>/my_constants.tmpl",
                    "output_file": "well_drill_constants.json",
                    "extra_data": "my_constants.yml",
                }
            ]
        case "template_render":
            updated_config_dict["forward_model"].insert(
                0,
                "template_render -i my_constants.yml -o well_drill_constants.json "
                "-t <CONFIG_PATH>/my_constants.tmpl",
            )

    config = everest_config_with_defaults(**updated_config_dict)

    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)

    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    # The data should have been loaded and passed through template to file.
    expected_file = (
        Path("everest_output")
        / "sim_output"
        / "batch_0"
        / "realization_0"
        / "perturbation_0"
        / "well_drill_constants.json"
    )
    assert expected_file.is_file()

    # Check expected contents of file
    contents = Path(expected_file).read_text(encoding="utf-8")
    assert contents == "VALUE1+VALUE2", (
        f'Expected contents: "VALUE1+VALUE2", found: {contents}'
    )


def test_that_install_template_raises_error_on_missing_render_template_fm_step():
    config = InstallTemplateConfig(
        template="some_file.tmpl", output_file="some_file.json"
    )

    with pytest.raises(
        KeyError, match=r"ERT forward model: template_render to be installed"
    ):
        config.to_ert_forward_model_step(
            control_names=["a", "b", "c"], installed_fm_steps={}, well_path="bla"
        )
