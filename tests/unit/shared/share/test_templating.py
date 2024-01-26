import json
import os
import pkgutil
import subprocess
from os.path import dirname
from typing import TYPE_CHECKING, cast

import jinja2
import pytest

from tests.utils import SOURCE_DIR

from ._import_from_location import import_from_location

if TYPE_CHECKING:
    from importlib.abc import FileLoader


# import template_render.py from ert/forward-models/templating/script
# package-data path which. These are kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.


template_render = import_from_location(
    "template_render",
    os.path.join(
        SOURCE_DIR,
        "src/ert/shared/share/ert/forward-models/templating/script/template_render.py",
    ),
)

render_template = template_render.render_template


def load_parameters():
    return template_render.load_data("parameters.json")


well_drill_tmpl = (
    'PROD1 takes value {{ well_drill.PROD1 }}, implying {{ "on" if well_drill.PROD1 >= 0.5 else "off" }}\n'  # noqa
    'PROD2 takes value {{ well_drill.PROD2 }}, implying {{ "on" if well_drill.PROD2 >= 0.5 else "off" }}\n'  # noqa
    "---------------------------------- \n"
    "{%- for well in well_drill.INJ %}\n"
    '{{ well.name }} takes value {{  well.value|round(1) }}, implying {{ "on" if  well.value >= 0.5 else "off"}}\n'  # noqa
    "{%- endfor %}"
)

optimal_template = "{{well_drill.values() | sum()}}"
dual_input = "{{ well_drill_north.PROD1 }} vs {{ well_drill_south.PROD1 }}"

mulitple_input_template = (
    "FILENAME\n"
    + "F1 {{parameters.key1.subkey1}}\n"
    + "OTH {{second.key1.subkey2}}\n"
    + "OTH_TEST {{third.key1.subkey1}}"
)

mulitple_input_template_no_param = (
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
    with open(prod_in, "w", encoding="utf-8") as fout:
        json.dump(prod_wells, fout)
    with open("parameters.json", "w", encoding="utf-8") as fout:
        json.dump(default_parameters, fout)
    with open("template_file", "w", encoding="utf-8") as fout:
        fout.write(well_drill_tmpl)

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

    with open(wells_in, "w", encoding="utf-8") as fout:
        json.dump(wells, fout)
    with open("parameters.json", "w", encoding="utf-8") as fout:
        json.dump(default_parameters, fout)
    with open(wells_tmpl, "w", encoding="utf-8") as fout:
        fout.write(well_drill_tmpl)

    render_template(wells_in, wells_tmpl, wells_out)
    expected_template_out = [
        "PROD1 takes value 0.2, implying off\n",
        "PROD2 takes value 0.4, implying off\n",
        "----------------------------------\n",
        "INJ1 takes value 0.8, implying on\n",
        "INJ2 takes value 0.6, implying on\n",
        "INJ3 takes value 0.4, implying off\n",
        "INJ4 takes value 0.2, implying off",
    ]

    with open(wells_out, encoding="utf-8") as fin:
        output = fin.readlines()

    assert output == expected_template_out


@pytest.mark.usefixtures("use_tmpdir")
def test_template_multiple_input():
    with open("template", "w", encoding="utf-8") as template_file:
        template_file.write(mulitple_input_template)

    with open("parameters.json", "w", encoding="utf-8") as json_file:
        json_file.write(json.dumps(default_parameters))

    with open("second.json", "w", encoding="utf-8") as json_file:
        parameters = {"key1": {"subkey2": 1400}}
        json.dump(parameters, json_file)
    with open("third.json", "w", encoding="utf-8") as json_file:
        parameters = {
            "key1": {
                "subkey1": 3000.22,
            }
        }
        json.dump(parameters, json_file)

    render_template(["second.json", "third.json"], "template", "out_file")

    with open("out_file", "r", encoding="utf-8") as parameter_file:
        expected_output = (
            "FILENAME\n" + "F1 1999.22\n" + "OTH 1400\n" + "OTH_TEST 3000.22"
        )

        assert parameter_file.read() == expected_output


@pytest.mark.usefixtures("use_tmpdir")
def test_no_parameters_json():
    with open("template", "w", encoding="utf-8") as template_file:
        template_file.write(mulitple_input_template_no_param)

    with open("not_the_standard_parameters.json", "w", encoding="utf-8") as json_file:
        json_file.write(json.dumps(default_parameters))

    with open("second.json", "w", encoding="utf-8") as json_file:
        parameters = {"key1": {"subkey2": 1400}}
        json.dump(parameters, json_file)
    with open("third.json", "w", encoding="utf-8") as json_file:
        parameters = {
            "key1": {
                "subkey1": 3000.22,
            }
        }
        json.dump(parameters, json_file)

    render_template(
        ["second.json", "third.json", "not_the_standard_parameters.json"],
        "template",
        "out_file",
    )

    with open("out_file", "r", encoding="utf-8") as parameter_file:
        expected_output = (
            "FILENAME\n" + "F1 1999.22\n" + "OTH 1400\n" + "OTH_TEST 3000.22"
        )

        assert parameter_file.read() == expected_output


@pytest.mark.usefixtures("use_tmpdir")
def test_template_executable():
    with open("template", "w", encoding="utf-8") as template_file:
        template_file.write(
            "FILENAME\n"
            + "F1 {{parameters.key1.subkey1}}\n"
            + "F2 {{other.key1.subkey1}}"
        )

    with open("parameters.json", "w", encoding="utf-8") as json_file:
        json_file.write(json.dumps(default_parameters))

    with open("other.json", "w", encoding="utf-8") as json_file:
        parameters = {
            "key1": {
                "subkey1": 200,
            }
        }
        json_file.write(json.dumps(parameters))

    params = (
        " --output_file out_file "
        "--template_file template "
        "--input_files other.json"
    )
    ert_shared_loader = cast("FileLoader", pkgutil.get_loader("ert.shared"))
    template_render_exec = (
        dirname(ert_shared_loader.get_filename())
        + "/share/ert/forward-models/templating/script/template_render.py"
    )

    subprocess.call(template_render_exec + params, shell=True, stdout=subprocess.PIPE)

    with open("out_file", "r", encoding="utf-8") as parameter_file:
        expected_output = "FILENAME\n" + "F1 1999.22\n" + "F2 200"
        assert parameter_file.read() == expected_output


@pytest.mark.usefixtures("use_tmpdir")
def test_load_parameters():
    with open("parameters.json", "w", encoding="utf-8") as json_file:
        json_file.write(json.dumps(default_parameters))

    assert load_parameters() == default_parameters
