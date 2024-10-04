import argparse
from copy import deepcopy as copy
from functools import partial
from os.path import exists, join
from typing import Any, Dict, Optional, Tuple

from ruamel.yaml import YAML
from seba_sqlite.database import Database as seba_db
from seba_sqlite.snapshot import SebaSnapshot

from everest.config import EverestConfig
from everest.config_file_loader import load_yaml
from everest.config_keys import ConfigKeys as CK


def _yaml_config(file_path: str, parser) -> Tuple[str, Optional[Dict[str, Any]]]:
    loaded_config = EverestConfig.load_file_with_argparser(file_path, parser)

    assert loaded_config is not None
    opt_folder = loaded_config.optimization_output_dir
    return opt_folder, load_yaml(file_path)


def _build_args_parser():
    arg_parser = argparse.ArgumentParser(
        description="Create new config file with updated controls "
        "from specified simulation batch number\n"
        "**Warning**: Previous simulation output folder will be overwritten"
        " if it was placed outside the everest output folder",
        usage="""everest branch <config_file> <new_config_file> -b #""",
    )
    arg_parser.add_argument(
        "input_config",
        help="The path to the everest configuration file",
        type=partial(_yaml_config, parser=arg_parser),
    )
    arg_parser.add_argument("output_config", help="The path to the new everest file")
    arg_parser.add_argument(
        "-b",
        "--batch",
        type=int,
        help="Batch id from which to retrieve control values",
        required=True,
    )
    return arg_parser


def opt_controls_by_batch(optimization_dir, batch):
    snapshot = SebaSnapshot(optimization_dir)
    for opt_data in snapshot.get_optimization_data():
        if opt_data.batch_id == batch:
            return opt_data.controls
    return None


def _updated_initial_guess(conf_controls, opt_controls):
    conf_controls = copy(conf_controls)

    for control in conf_controls:
        control.pop(CK.INITIAL_GUESS, None)
        control_name = "{}_".format(control[CK.NAME])
        batch_controls = {
            key.split(control_name)[-1]: val
            for key, val in opt_controls.items()
            if control_name in key
        }

        for variable in control[CK.VARIABLES]:
            var_index = variable.get(CK.INDEX, None)

            if var_index is not None:
                opt_control_name = "{}-{}".format(variable[CK.NAME], var_index)
            else:
                opt_control_name = variable[CK.NAME]

            opt_control_val = batch_controls.get(opt_control_name)

            if opt_control_val is None:
                print(
                    "No generated optimization control value found for"
                    " control {} index {}".format(variable[CK.NAME], var_index)
                )
                return None
            else:
                variable[CK.INITIAL_GUESS] = opt_control_val

    return conf_controls


def config_branch_entry(args=None):
    parser = _build_args_parser()
    options = parser.parse_args(args)
    optimization_dir, yml_config = options.input_config

    db_path = join(optimization_dir, seba_db.FILENAME)
    if not exists(db_path):
        parser.error("Optimization source {} not found".format(db_path))

    opt_controls = opt_controls_by_batch(optimization_dir, options.batch)
    if opt_controls is None:
        parser.error("Batch {} not present in optimization data".format(options.batch))

    yml_config[CK.CONTROLS] = _updated_initial_guess(
        conf_controls=yml_config[CK.CONTROLS], opt_controls=opt_controls
    )

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)  # pylint: disable=not-callable
    yaml.preserve_quotes = True
    with open(options.output_config, "w", encoding="utf-8") as f:
        yaml.dump(yml_config, f)
    print("New config file {} created.".format(options.output_config))


if __name__ == "__main__":
    config_branch_entry()
