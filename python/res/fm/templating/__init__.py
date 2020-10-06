from .template_render import render_template
from .template_render import load_data


def load_parameters():
    return load_data("parameters.json")
