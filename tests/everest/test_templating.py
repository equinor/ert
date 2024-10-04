import json
import os
import subprocess

import pytest
from ruamel.yaml import YAML

import everest
from everest.config import EverestConfig

TMPL_CONFIG_FILE = "config.yml"
TMPL_WELL_DRILL_FILE = os.path.join("templates", "well_drill_info.tmpl")
TMPL_DUAL_INPUT_FILE = os.path.join("templates", "dual_input.tmpl")

MATH_CONFIG_FILE = "config_minimal.yml"


def test_render_invalid(copy_template_test_data_to_tmp):
    render = everest.jobs.templating.render

    prod_wells = {"PROD%d" % idx: 0.3 * idx for idx in range(4)}
    prod_in = "well_drill_prod.json"
    with open(prod_in, "w", encoding="utf-8") as fout:
        json.dump(prod_wells, fout)

    wells_out = "wells.out"

    with pytest.raises(TypeError):
        render(None, TMPL_WELL_DRILL_FILE, wells_out)

    with pytest.raises(ValueError):
        render(2 * prod_in, TMPL_WELL_DRILL_FILE, wells_out)

    with pytest.raises(TypeError):
        render(prod_in, None, wells_out)

    with pytest.raises(ValueError):
        render(prod_in, TMPL_WELL_DRILL_FILE + "nogo", wells_out)

    with pytest.raises(TypeError):
        render(prod_in, TMPL_WELL_DRILL_FILE, None)


def test_render(copy_template_test_data_to_tmp):
    render = everest.jobs.templating.render

    wells = {"PROD%d" % idx: 0.2 * idx for idx in range(1, 5)}
    wells.update({"INJ%d" % idx: 1 - 0.2 * idx for idx in range(1, 5)})
    wells_in = "well_drill.json"
    with open(wells_in, "w", encoding="utf-8") as fout:
        json.dump(wells, fout)

    wells_out = "wells.out"
    render(wells_in, TMPL_WELL_DRILL_FILE, wells_out)

    with open(wells_out, encoding="utf-8") as fin:
        output = fin.readlines()

    for idx, line in enumerate(output):
        split = line.split(" ")
        if len(split) == 1:
            assert idx == 2
            assert line == "----------------------------------\n"
        else:
            on_off = "on" if wells[split[0]] >= 0.5 else "off"
            expected_string = "{} takes value {}, implying {}\n".format(
                split[0], wells[split[0]], on_off
            )
            if idx == len(output) - 1:
                expected_string = expected_string[:-1]
            assert expected_string == line


def test_render_multiple_input(copy_template_test_data_to_tmp):
    render = everest.jobs.templating.render

    wells_north = {"PROD%d" % idx: 0.2 * idx for idx in range(1, 5)}
    wells_north_in = "well_drill_north.json"
    with open(wells_north_in, "w", encoding="utf-8") as fout:
        json.dump(wells_north, fout)

    wells_south = {"PROD%d" % idx: 1 - 0.2 * idx for idx in range(1, 5)}
    wells_south_in = "well_drill_south.json"
    with open(wells_south_in, "w", encoding="utf-8") as fout:
        json.dump(wells_south, fout)

    wells_out = "sub_folder/wells.out"
    render((wells_north_in, wells_south_in), TMPL_DUAL_INPUT_FILE, wells_out)

    with open(wells_out, encoding="utf-8") as fin:
        output = fin.readlines()

    assert output == ["0.2 vs 0.8"]


def test_render_executable(copy_template_test_data_to_tmp):
    assert os.access(everest.jobs.render, os.X_OK)

    # Dump input
    wells_north = {"PROD%d" % idx: 0.2 * idx for idx in range(1, 5)}
    wells_north_in = "well_drill_north.json"
    with open(wells_north_in, "w", encoding="utf-8") as fout:
        json.dump(wells_north, fout)

    wells_south = {"PROD%d" % idx: 1 - 0.2 * idx for idx in range(1, 5)}
    wells_south_in = "well_drill_south.json"
    with open(wells_south_in, "w", encoding="utf-8") as fout:
        json.dump(wells_south, fout)

    # Format command
    output_file = "render_out"
    cmd_fmt = "{render} --output {fout} --template {tmpl} --input_files {fin}"
    cmd = cmd_fmt.format(
        render=everest.jobs.render,
        tmpl=TMPL_DUAL_INPUT_FILE,
        fout=output_file,
        fin=" ".join((wells_north_in, wells_south_in)),
    )

    subprocess.check_call(cmd, shell=True)

    # Verify result
    with open(output_file, encoding="utf-8") as fout:
        assert "\n".join(fout.readlines()) == "0.2 vs 0.8"


@pytest.mark.integration_test
def test_install_template(copy_template_test_data_to_tmp):
    config = EverestConfig.load_file(TMPL_CONFIG_FILE)
    workflow = everest.suite._EverestWorkflow(config)
    workflow.start_optimization()


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
    everest.jobs.templating.render(
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
def test_user_specified_data_n_template(copy_math_func_test_data_to_tmp):
    """
    Ensure that a user specifying a data resource and an installed_template
    with "extra_data", the results of that template will be passed to the
    directory for each consecutive simulation
    """

    config = EverestConfig.load_file(MATH_CONFIG_FILE)

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
            "install_templates": [
                {
                    "template": "<CONFIG_PATH>/my_constants.tmpl",
                    "output_file": "well_drill_constants.json",
                    "extra_data": "my_constants.yml",
                }
            ],
        }
    )

    config = EverestConfig.with_defaults(**updated_config_dict)

    workflow = everest.suite._EverestWorkflow(config)
    assert workflow is not None

    workflow.start_optimization()

    # The data should have been loaded and passed through template to file.
    expected_file = os.path.join(
        "everest_output",
        "sim_output",
        "batch_0",
        "geo_realization_0",
        "simulation_1",
        "well_drill_constants.json",
    )
    assert os.path.isfile(expected_file)

    # Check expected contents of file
    with open(expected_file, "r", encoding="utf-8") as f:
        contents = f.read()
    assert (
        contents == "VALUE1+VALUE2"
    ), 'Expected contents: "VALUE1+VALUE2", found: {}'.format(contents)
