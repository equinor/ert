import importlib
import json
import os
import subprocess
from pathlib import Path

import jinja2
import pytest

from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import template_render.py from ert/forward_models package-data path which.
# These are kept out of the ert package to avoid the overhead of importing ert.
# This is necessary as these may be invoked as a subprocess on each
# realization.


template_render = import_from_location(
    "template_render",
    os.path.join(
        SOURCE_DIR,
        "src/ert/resources/forward_models/template_render.py",
    ),
)

render_template = template_render.render_template


def load_parameters():
    return template_render.load_data("parameters.json")


well_drill_tmpl = (
    "PROD1 takes value {{ well_drill.PROD1 }}, "
    'implying {{ "on" if well_drill.PROD1 >= 0.5 else "off" }}\n'
    "PROD2 takes value {{ well_drill.PROD2 }}, "
    'implying {{ "on" if well_drill.PROD2 >= 0.5 else "off" }}\n'
    "---------------------------------- \n"
    "{%- for well in well_drill.INJ %}\n"
    "{{ well.name }} takes value {{  well.value|round(1) }}, "
    'implying {{ "on" if  well.value >= 0.5 else "off"}}\n'
    "{%- endfor %}"
)

optimal_template = "{{well_drill.values() | sum()}}"
dual_input = "{{ well_drill_north.PROD1 }} vs {{ well_drill_south.PROD1 }}"

multiple_input_template = (
    "FILENAME\n"
    + "F1 {{parameters.key1.subkey1}}\n"
    + "OTH {{second.key1.subkey2}}\n"
    + "OTH_TEST {{third.key1.subkey1}}"
)

multiple_input_template_no_param = (
    "FILENAME\n"
    + "F1 {{not_the_standard_parameters.key1.subkey1}}\n"
    + "OTH {{second.key1.subkey2}}\n"
    + "OTH_TEST {{third.key1.subkey1}}"
)

default_parameters = {
    "key1": {"subkey1": 1999.22, "subkey2": 200},
    "key2": {"subkey1": 300},
}


@pytest.mark.usefixtures("use_tmpdir")
def test_render_invalid():
    prod_wells = {f"PROD{idx}": 0.3 * idx for idx in range(4)}
    prod_in = "well_drill.json"
    Path(prod_in).write_text(json.dumps(prod_wells), encoding="utf-8")
    Path("parameters.json").write_text(json.dumps(default_parameters), encoding="utf-8")
    Path("template_file").write_text(well_drill_tmpl, encoding="utf-8")

    wells_out = "wells.out"

    # undefined template elements
    with pytest.raises(jinja2.exceptions.UndefinedError):
        render_template(None, "template_file", wells_out)

    # file not found
    with pytest.raises(ValueError):
        render_template(2 * prod_in, "template_file", wells_out)

    # no template file
    with pytest.raises(TypeError):
        render_template(prod_in, None, wells_out)

    # templatefile not found
    with pytest.raises(ValueError):
        render_template(prod_in, "template_file" + "nogo", wells_out)

    # no output file
    with pytest.raises(TypeError):
        render_template(prod_in, "template_file", None)


@pytest.mark.usefixtures("use_tmpdir")
def test_render():
    wells = {f"PROD{idx}": 0.2 * idx for idx in range(1, 5)}
    wells.update(
        {"INJ": [{"name": f"INJ{idx}", "value": 1 - 0.2 * idx} for idx in range(1, 5)]}
    )
    wells_in = "well_drill.json"
    wells_tmpl = "well_drill_tmpl"
    wells_out = "wells.out"

    Path(wells_in).write_text(json.dumps(wells), encoding="utf-8")
    Path("parameters.json").write_text(json.dumps(default_parameters), encoding="utf-8")
    Path(wells_tmpl).write_text(well_drill_tmpl, encoding="utf-8")

    render_template(wells_in, wells_tmpl, wells_out)
    expected_template_out = [
        "PROD1 takes value 0.2, implying off",
        "PROD2 takes value 0.4, implying off",
        "----------------------------------",
        "INJ1 takes value 0.8, implying on",
        "INJ2 takes value 0.6, implying on",
        "INJ3 takes value 0.4, implying off",
        "INJ4 takes value 0.2, implying off",
    ]

    assert (
        Path(wells_out).read_text(encoding="utf-8").splitlines()
        == expected_template_out
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_template_multiple_input():
    Path("template").write_text(multiple_input_template, encoding="utf-8")

    Path("parameters.json").write_text(json.dumps(default_parameters), encoding="utf-8")

    Path("second.json").write_text(
        json.dumps({"key1": {"subkey2": 1400}}), encoding="utf-8"
    )

    Path("third.json").write_text(
        json.dumps(
            {
                "key1": {
                    "subkey1": 3000.22,
                }
            }
        ),
        encoding="utf-8",
    )

    render_template(["second.json", "third.json"], "template", "out_file")

    assert (
        Path("out_file").read_text(encoding="utf-8")
        == "FILENAME\n" + "F1 1999.22\n" + "OTH 1400\n" + "OTH_TEST 3000.22"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_no_parameters_json():
    Path("template").write_text(multiple_input_template_no_param, encoding="utf-8")

    Path("not_the_standard_parameters.json").write_text(
        json.dumps(default_parameters), encoding="utf-8"
    )

    Path("second.json").write_text(
        json.dumps({"key1": {"subkey2": 1400}}), encoding="utf-8"
    )

    Path("third.json").write_text(
        json.dumps(
            {
                "key1": {
                    "subkey1": 3000.22,
                }
            }
        ),
        encoding="utf-8",
    )

    render_template(
        ["second.json", "third.json", "not_the_standard_parameters.json"],
        "template",
        "out_file",
    )

    assert (
        Path("out_file").read_text(encoding="utf-8")
        == "FILENAME\n" + "F1 1999.22\n" + "OTH 1400\n" + "OTH_TEST 3000.22"
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.integration_test
def test_template_executable():
    Path("template").write_text(
        "FILENAME\nF1 {{parameters.key1.subkey1}}\nF2 {{other.key1.subkey1}}",
        encoding="utf-8",
    )

    Path("parameters.json").write_text(json.dumps(default_parameters), encoding="utf-8")

    Path("other.json").write_text(
        json.dumps(
            {
                "key1": {
                    "subkey1": 200,
                }
            }
        ),
        encoding="utf-8",
    )

    params = " --output_file out_file --template_file template --input_files other.json"
    template_render_exec = str(
        Path(importlib.util.find_spec("ert").origin).parent
        / "resources/forward_models/template_render.py"
    )

    subprocess.call(template_render_exec + params, shell=True, stdout=subprocess.PIPE)

    assert (
        Path("out_file").read_text(encoding="utf-8")
        == "FILENAME\n" + "F1 1999.22\n" + "F2 200"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_load_parameters():
    Path("parameters.json").write_text(json.dumps(default_parameters), encoding="utf-8")

    assert load_parameters() == default_parameters
