import json
import os
from pathlib import Path

import jinja2
import pytest
from ruamel.yaml import YAML

import everest
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import ErtPluginContext
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig, InstallTemplateConfig
from tests.ert.unit_tests.resources._import_from_location import import_from_location
from tests.ert.utils import SOURCE_DIR

CONFIG = {
    "wells": [
        {"name": "PROD1", "drill_time": 30},
        {"name": "PROD2", "drill_time": 60},
    ],
    "controls": [
        {
            "name": "well_drill",
            "type": "well_control",
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
DUAL_INPUT_TMPL = "{{ well_drill_north.PROD1 }} vs {{ well_drill_south.PROD1 }}"


template_render = import_from_location(
    "template_render",
    os.path.join(
        SOURCE_DIR,
        "src/ert/resources/forward_models/template_render.py",
    ),
)


def test_render_invalid(change_to_tmpdir):
    template_file = "well_drill_info.tmpl"
    with open(template_file, "w", encoding="utf-8") as fp:
        fp.write(WELL_DRILL_TMPL)

    render = template_render.render_template

    prod_wells = {f"PROD{idx:d}": 0.3 * idx for idx in range(4)}
    prod_in = "well_drill_prod.json"
    with open(prod_in, "w", encoding="utf-8") as fout:
        json.dump(prod_wells, fout)

    wells_out = "wells.out"

    with pytest.raises(jinja2.exceptions.UndefinedError):
        render(None, template_file, wells_out)

    with pytest.raises(ValueError):
        render(2 * prod_in, template_file, wells_out)

    with pytest.raises(TypeError):
        render(prod_in, None, wells_out)

    with pytest.raises(ValueError):
        render(prod_in, template_file + "nogo", wells_out)

    with pytest.raises(TypeError):
        render(prod_in, template_file, None)


def test_render(change_to_tmpdir):
    template_file = "well_drill_info.tmpl"
    with open(template_file, "w", encoding="utf-8") as fp:
        fp.write(WELL_DRILL_TMPL)

    render = template_render.render_template

    wells = {f"PROD{idx:d}": 0.2 * idx for idx in range(1, 5)}
    wells.update({f"INJ{idx:d}": 1 - 0.2 * idx for idx in range(1, 5)})
    wells_in = "well_drill.json"
    with open(wells_in, "w", encoding="utf-8") as fout:
        json.dump(wells, fout)

    wells_out = "wells.out"
    render(wells_in, template_file, wells_out)

    with open(wells_out, encoding="utf-8") as fin:
        output = fin.readlines()

    for idx, line in enumerate(output):
        split = line.split(" ")
        if len(split) == 1:
            assert idx == 2
            assert line == "----------------------------------\n"
        else:
            on_off = "on" if wells[split[0]] >= 0.5 else "off"
            expected_string = (
                f"{split[0]} takes value {wells[split[0]]}, implying {on_off}\n"
            )
            if idx == len(output) - 1:
                expected_string = expected_string[:-1]
            assert expected_string == line


def test_render_multiple_input(change_to_tmpdir):
    template_file = "dual_input.tmpl"
    with open(template_file, "w", encoding="utf-8") as fp:
        fp.write(DUAL_INPUT_TMPL)

    render = template_render.render_template

    wells_north = {f"PROD{idx:d}": 0.2 * idx for idx in range(1, 5)}
    wells_north_in = "well_drill_north.json"
    with open(wells_north_in, "w", encoding="utf-8") as fout:
        json.dump(wells_north, fout)

    wells_south = {f"PROD{idx:d}": 1 - 0.2 * idx for idx in range(1, 5)}
    wells_south_in = "well_drill_south.json"
    with open(wells_south_in, "w", encoding="utf-8") as fout:
        json.dump(wells_south, fout)

    wells_out = "sub_folder/wells.out"
    render((wells_north_in, wells_south_in), template_file, wells_out)

    with open(wells_out, encoding="utf-8") as fin:
        output = fin.readlines()

    assert output == ["0.2 vs 0.8"]


@pytest.mark.integration_test
def test_install_template(change_to_tmpdir):
    with open("config.yml", "w", encoding="utf-8") as fp:
        YAML(typ="safe", pure=True).dump(CONFIG, fp)
    template_file = "well_drill_info.tmpl"
    with open(template_file, "w", encoding="utf-8") as fp:
        fp.write(WELL_DRILL_TMPL)
    template_file = "the_optimal_template.tmpl"
    with open(template_file, "w", encoding="utf-8") as fp:
        fp.write(THE_OPTIMAL_TEMPLATE_TMPL)
    config = EverestConfig.load_file("config.yml")
    with ErtPluginContext() as runtime_plugins:
        run_model = EverestRunModel.create(config, runtime_plugins=runtime_plugins)
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
    with open(data_file, "w", encoding="utf-8") as fout:
        json.dump(well_order, fout)

    output_file = "well_order_list.json"
    template_render.render_template(
        data_file,
        order_tmpl,
        output_file,
    )

    with open(output_file, encoding="utf-8") as fin:
        order = json.load(fin)

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
    yaml = YAML(typ="safe", pure=True)
    with open("my_constants.yml", "w", encoding="utf-8") as f:
        yaml.dump({"CONST1": "VALUE1", "CONST2": "VALUE2"}, f)

    # Write out the template to which takes the constants above
    with open("my_constants.tmpl", "w", encoding="utf-8") as f:
        f.write("{{ my_constants.CONST1 }}+{{ my_constants.CONST2 }}")

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

    config = EverestConfig.with_defaults(**updated_config_dict)

    with ErtPluginContext() as runtime_plugins:
        run_model = EverestRunModel.create(config, runtime_plugins=runtime_plugins)
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
