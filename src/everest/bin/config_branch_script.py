import argparse
from copy import deepcopy as copy
from functools import partial
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from everest.bin.utils import setup_logging
from everest.config import EverestConfig
from everest.config_file_loader import load_yaml
from everest.everest_storage import EverestStorage


def _yaml_config(
    file_path: str, parser: argparse.ArgumentParser
) -> tuple[str, str, dict[str, Any] | None]:
    loaded_config = EverestConfig.load_file_with_argparser(file_path, parser)
    assert loaded_config is not None
    opt_folder = loaded_config.optimization_output_dir
    return file_path, opt_folder, load_yaml(file_path)


def _build_args_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Create new config file with updated controls "
        "from specified simulation batch number\n"
        "**Warning**: Previous simulation output folder will be overwritten"
        " if it was placed outside the everest output folder",
        usage="""everest branch <config_file> <new_config_file> -b #""",
    )
    arg_parser.add_argument(
        "config",
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


def opt_controls_by_batch(optimization_dir: str, batch: int) -> dict[str, Any] | None:
    storage = EverestStorage(Path(optimization_dir))
    storage.read_from_output_dir()

    assert storage.data is not None
    assert storage.data.controls is not None
    control_names = storage.data.controls["control_name"]
    function_batch = next(
        (b for b in storage.data.batches_with_function_results if b.batch_id == batch),
        None,
    )

    if function_batch:
        # All geo-realizations should have the same unperturbed control values per batch
        # hence it does not matter which realization we select the controls for
        return function_batch.realization_controls.select(
            control_names.to_list()
        ).to_dicts()[0]

    return None


def _updated_initial_guess(
    conf_controls: list[dict[str, Any]], opt_controls: dict[str, Any]
) -> list[dict[str, Any]] | None:
    conf_controls = copy(conf_controls)

    for control in conf_controls:
        control_name = f"{control['name']}."
        control.pop("initial_guess", None)
        batch_controls = {
            key.split(control_name)[-1]: val
            for key, val in opt_controls.items()
            if control_name in key
        }

        for variable in control["variables"]:
            var_index = variable.get("index", None)

            if var_index is not None:
                opt_control_name = f"{variable['name']}.{var_index}"
            else:
                opt_control_name = variable["name"]

            opt_control_val = batch_controls.get(opt_control_name)

            if opt_control_val is None:
                print(
                    "No generated optimization control value found for"
                    f" control {variable['name']} index {var_index}"
                )
                return None
            else:
                variable["initial_guess"] = opt_control_val

    return conf_controls


def config_branch_entry(args: list[str] | None = None) -> None:
    parser = _build_args_parser()
    options = parser.parse_args(args)
    with setup_logging(options):
        _, optimization_dir, yml_config = options.config

        EverestStorage.check_for_deprecated_seba_storage(optimization_dir)

        opt_controls = opt_controls_by_batch(optimization_dir, options.batch)
        if opt_controls is None:
            parser.error(f"Batch {options.batch} not present in optimization data")

        yml_config["controls"] = _updated_initial_guess(
            conf_controls=yml_config["controls"], opt_controls=opt_controls
        )

        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.preserve_quotes = True
        with open(options.output_config, "w", encoding="utf-8") as f:
            yaml.dump(yml_config, f)
        print(f"New config file {options.output_config} created.")


if __name__ == "__main__":
    config_branch_entry()
