#!/usr/bin/env python

import logging
import os
from typing import Any, Dict, List, Optional

import jinja2
from ruamel.yaml import YAML, YAMLError

from .config_keys import ConfigKeys

# Since YAML interprets '{' as start of a dict, we need to prefix it with
# something to make reading possible.
#
# We chose to use 'r{{' and '}}', and 'r{%' and '%}' as the template rendering
# delimiters, '<' and '<%' are also good options.  These should be somewhat
# consistent with the use in the template system used outside of the config file
# as well.

BLOCK_START_STRING = "r{%"  # end string remains as '}'
VARIABLE_START_STRING = "r{{"  # end string remains as '}}'


# Jinja vars which should NOT be included in definitions portion of config.
ERT_CONFIG_TEMPLATES = {
    "realization": "GEO_ID",
    "runpath_file": "RUNPATH_FILE",
}


def load_yaml(file_name: str) -> Optional[Dict[str, Any]]:
    with open(file_name, "r", encoding="utf-8") as input_file:
        input_data: List[str] = input_file.readlines()
        try:
            yaml = YAML()
            yaml.preserve_quotes = True
            return yaml.load("".join(input_data))
        except YAMLError as exc:
            if hasattr(exc, "problem_mark"):
                mark = exc.problem_mark
                raise YAMLError(
                    str(exc)
                    + "\nError in line: {}\n {}^)".format(
                        input_data[mark.line], " " * mark.column
                    )
                ) from exc

        return None


def _get_definitions(configuration, configpath):
    defs = {}
    if configuration:
        if ConfigKeys.DEFINITIONS not in configuration:
            msg = "No {} node found in configuration file"
            logging.debug(msg.format(ConfigKeys.DEFINITIONS))
        else:
            defs = configuration.get(ConfigKeys.DEFINITIONS, {})

        for key, val in ERT_CONFIG_TEMPLATES.items():
            if key in defs:
                logging.warn(
                    "Internal key {k} specified by user as {u}. "
                    "Overriding as {v}".format(k=key, u=defs[key], v=val)
                )
            defs[key] = "<{}>".format(val)  # ert uses <GEO_ID> as format
    else:
        logging.warn("Empty configuration file provided!")

    # If user didn't define a config path, we can insert it here.
    defs["configpath"] = defs.get("configpath", configpath)

    # If user didn't define a eclbase arg for eclipse100, we insert it.
    defs[ConfigKeys.ECLBASE] = defs.get(ConfigKeys.ECLBASE, "eclipse/ECL")

    return defs


def _os():
    """Return an object whose properties are the users environment variables.

    For example, calling os.USER returns the username, os.HOSTNAME returns the
    hostname (according to the environment variable).  It is used by Jinja to
    substitute {{ os.VAR }} with the corresponding value os.environ[VAR].

    """

    class Os(object):
        pass

    x = Os()
    x.__dict__.update(os.environ)
    return x


def _render_definitions(definitions, jinja_env):
    # pylint: disable=unnecessary-lambda-assignment
    render = lambda s, d: jinja_env.from_string(s).render(**d)
    for key in definitions:
        if not isinstance(definitions[key], str):
            continue

        for _idx in range(len(definitions) + 1):
            new_val = render(definitions[key], definitions)
            if definitions[key] != new_val:
                definitions[key] = new_val
            else:
                break

        if VARIABLE_START_STRING in definitions[key]:
            raise ValueError(
                """Circular dependencies in definitions. Please """
                """resolve using everlint."""
            )


def yaml_file_to_substituted_config_dict(config_path: str) -> Dict[str, Any]:
    configuration = load_yaml(config_path)

    definitions = _get_definitions(
        configuration=configuration,
        configpath=os.path.dirname(os.path.abspath(config_path)),
    )
    definitions["os"] = _os()  # update definitions with os namespace
    with open(config_path, "r", encoding="utf-8") as f:
        txt = "".join(f.readlines())
    jenv = jinja2.Environment(
        block_start_string=BLOCK_START_STRING,
        variable_start_string=VARIABLE_START_STRING,
    )

    _render_definitions(definitions, jenv)

    # Replace in definitions
    config = jenv.from_string(txt).render(**definitions)

    # Load the config with definitions again as yaml
    yaml = YAML(typ="safe", pure=True).load(config)

    if not isinstance(yaml, Dict):
        yaml = {}

    # Inject config path
    yaml[ConfigKeys.CONFIGPATH] = config_path
    return yaml
