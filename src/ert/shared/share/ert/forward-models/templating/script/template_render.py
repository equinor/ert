#!/usr/bin/env python
import argparse
import json
import os

import jinja2
import yaml

DEFAULT_GEN_KW_EXPORT_NAME = "parameters"


def load_data(filename):
    """Will try to load data from @filename first as yaml, and if that fails,
    as json. If both fail, a ValueError with both of the error messages will be
    raised.
    """
    yaml_err = ""
    json_err = ""
    with open(filename, encoding="utf-8") as fin:
        try:
            return yaml.safe_load(fin)
        except yaml.YAMLError as err:
            yaml_err = str(err)
            pass

        try:
            return json.load(fin)
        except yaml.YAMLError as err:
            json_err = str(err)
            pass

    err_msg = "%s is neither yaml (err_msg=%s) nor json (err_msg=%s)"
    raise IOError(err_msg % (filename, str(yaml_err), str(json_err)))


def _load_template(template_path):
    path, filename = os.path.split(template_path)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or "./")
    ).get_template(filename)


def _generate_file_namespace(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _load_input(input_files):
    """
    Loads input files (JSON or YAML) and returns the content as dict.
    """
    data = {}
    for input_file in input_files:
        input_namespace = _generate_file_namespace(input_file)
        data[input_namespace] = load_data(input_file)

    return data


def _assert_input(input_files, template_file, output_file):
    """
    validates input for template rendering.
    Throws ValueError if input files or template file is not found.
    Throws TypeError if output_file is not a string.
    """
    for input_file in input_files:
        if not os.path.isfile(input_file):
            raise ValueError(f"Input file: {input_file}, does not exist..")

    if not os.path.isfile(template_file):
        raise ValueError(f"Template file: {template_file}, does not exist..")

    if not isinstance(output_file, str):
        raise TypeError("Expected output path to be a string")


def render_template(input_files, template_file, output_file):
    """
    Will render a jinja2 template file with the parameters given
    :param input_files: parameters as a list of JSON or YAML files
    :param template_file: template file in jinja2 format
    :param output_file: output destination for the rendered template file
    """
    if isinstance(input_files, str) and input_files:
        input_files = (input_files,)

    all_input_files = ()

    gen_kw_export_path = DEFAULT_GEN_KW_EXPORT_NAME + ".json"
    if os.path.isfile(gen_kw_export_path):
        all_input_files += (gen_kw_export_path,)

    if input_files:
        all_input_files += tuple(input_files)

    _assert_input(all_input_files, template_file, output_file)

    template = _load_template(template_file)
    data = _load_input(all_input_files)
    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write(template.render(**data))


def _build_argument_parser():
    description = """
Loads the data from each file ("some/path/filename.xxx") in INPUT_FILES
and exposes it as the variable "filename". It then loads the Jinja2
template TEMPLATE_FILE and dumps the rendered result to OUTPUT.

Example:
Given an input file my_input.json:

{
    my_variable: my_value
}

And a template file tmpl.jinja:

This is written in my file together with {{my_input.my_variable}}

This job will produce an output file:

This is written in my file together with my_value
"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output_file",
        "-o",
        required=True,
        help="the output file",
    )
    parser.add_argument(
        "--template_file",
        "-t",
        required=True,
        help="the jinja2 template file",
    )
    parser.add_argument(
        "--input_files",
        "-i",
        nargs="+",
        help="list of json and yaml input files",
    )
    return parser


if __name__ == "__main__":
    arg_parser = _build_argument_parser()
    args = arg_parser.parse_args()

    render_template(
        args.input_files,
        args.template_file,
        args.output_file,
    )
