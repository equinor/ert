import os

import jinja2

import everest


def _load_template(template_path):
    path, filename = os.path.split(template_path)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or "./")
    ).get_template(filename)


def _generate_file_namespace(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _load_input(input_files):
    data = {}
    for input_file in input_files:
        input_namespace = _generate_file_namespace(input_file)
        data[input_namespace] = everest.jobs.io.load_data(input_file)

    return data


def _assert_input(input_files, template_file, output_file):
    for input_file in input_files:
        if not os.path.isfile(input_file):
            raise ValueError("Input file: %s, does not exist.." % input_file)

    if not os.path.isfile(template_file):
        raise ValueError("Template file: %s, does not exist.." % template_file)

    if not isinstance(output_file, str):
        raise TypeError("Expected output path to be a string")


def render(input_files, template_file, output_file):
    if isinstance(input_files, str):
        input_files = (input_files,)
    _assert_input(input_files, template_file, output_file)

    template = _load_template(template_file)
    data = _load_input(input_files)

    with everest.jobs.io.safe_open(output_file, "w") as fout:
        fout.write(template.render(**data))
