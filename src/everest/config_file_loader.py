#!/usr/bin/env python

import logging
import os
import re
from io import StringIO
from typing import Any

import jinja2
from ruamel.yaml import YAML, YAMLError

from everest.strings import EVEREST

# Since YAML interprets '{' as start of a dict, we need to prefix it with
# something to make reading possible.
#
# We chose to use 'r{{' and '}}', and 'r{%' and '%}' as the template rendering
# delimiters, '<' and '<%' are also good options.  These should be somewhat
# consistent with the use in the template system used outside of the config file
# as well.

BLOCK_START_STRING = "r{%"  # end string remains as '}'
VARIABLE_START_STRING = "r{{"  # end string remains as '}}'
SUBSTITUTION_PATTERN = r"(r\{\{.*?\}\})"

# Jinja vars which should NOT be included in definitions portion of config.
ERT_CONFIG_TEMPLATES = {
    "realization": "GEO_ID",
    "runpath_file": "RUNPATH_FILE",
}


def load_yaml(file_name: str, safe: bool = False) -> dict[str, Any]:
    with open(file_name, encoding="utf-8") as input_file:
        input_data: list[str] = input_file.readlines()
        try:
            yaml = YAML(typ="safe", pure=True) if safe else YAML()
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
            else:
                raise YAMLError(str(exc)) from exc


def _get_definitions(
    configuration: dict[str, Any] | None, configpath: str
) -> dict[str, Any]:
    defs = {}
    if configuration:
        if "definitions" not in configuration:
            msg = "No {} node found in configuration file"
            logging.getLogger(EVEREST).debug(msg.format("definitions"))
        else:
            defs = configuration.get("definitions", {})

        for key, val in ERT_CONFIG_TEMPLATES.items():
            if key in defs:
                logging.getLogger(EVEREST).warning(
                    f"Internal key {key} specified by user as {defs[key]}. "
                    f"Overriding as {val}"
                )
            defs[key] = f"<{val}>"  # ert uses <GEO_ID> as format
    else:
        logging.getLogger(EVEREST).warning("Empty configuration file provided!")

    # If user didn't define a config path, we can insert it here.
    defs["configpath"] = defs.get("configpath", configpath)

    # If user didn't define a eclbase arg for eclipse100, we insert it.
    defs["eclbase"] = defs.get("eclbase", "eclipse/ECL")

    return defs


class Os:
    pass


def _os() -> Os:
    """Return an object whose properties are the users environment variables.

    For example, calling os.USER returns the username, os.HOSTNAME returns the
    hostname (according to the environment variable).  It is used by Jinja to
    substitute {{ os.VAR }} with the corresponding value os.environ[VAR].

    """

    x = Os()
    x.__dict__.update(os.environ)
    return x


def _render_definitions(
    definitions: dict[str, Any], jinja_env: jinja2.Environment
) -> None:
    def render(s: str, d: dict[str, Any]) -> str:
        return jinja_env.from_string(s).render(**d)

    for key in definitions:  # noqa: PLC0206
        if not isinstance(definitions[key], str):
            continue

        for _idx in range(len(definitions) + 1):
            new_val = render(definitions[key], definitions)
            if definitions[key] != new_val:
                definitions[key] = new_val
            else:
                break

        if VARIABLE_START_STRING in definitions[key]:
            raise ValueError("Circular dependencies in definitions.")


def yaml_file_to_substituted_config_dict(config_path: str) -> dict[str, Any]:
    configuration = load_yaml(config_path, safe=True)
    if configuration is None:
        return {}

    yaml = YAML()
    buffer = StringIO()
    yaml.dump(configuration, buffer)
    txt = buffer.getvalue()
    buffer.close()

    definitions = _get_definitions(
        configuration=configuration,
        configpath=os.path.dirname(os.path.abspath(config_path)),
    )
    definitions["os"] = _os()  # update definitions with os namespace

    jenv = jinja2.Environment(
        block_start_string=BLOCK_START_STRING,
        variable_start_string=VARIABLE_START_STRING,
    )

    undefined = [
        s
        for s in re.findall(SUBSTITUTION_PATTERN, txt)
        if not jenv.from_string(s).render(**definitions)
    ]

    if undefined:
        more_than_one = len(undefined) > 1
        raise ValueError(
            f"The following key{'s' if more_than_one else ''} "
            f"{'are' if more_than_one else 'is'} missing: {undefined} "
            f"in the definitions section"
        )

    _render_definitions(definitions, jenv)

    # Replace in definitions
    config = jenv.from_string(txt).render(**definitions)

    # Load the config with definitions again as yaml
    yaml = YAML(typ="safe", pure=True).load(config)

    if not isinstance(yaml, dict):
        return {"config_path": config_path}
    else:
        # Inject config path
        yaml["config_path"] = config_path
        return yaml
