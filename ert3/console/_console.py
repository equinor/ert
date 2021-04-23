import argparse
import pathlib
import shutil
import sys

from typing import Any, List, Union
from pathlib import Path

import pkg_resources as pkg
import yaml

import ert3

from ert3.config import EnsembleConfig, StagesConfig, ExperimentConfig


_ERT3_DESCRIPTION = (
    "ert3 is an ensemble-based tool for uncertainty studies.\n"
    "\nWARNING: the tool is currently in an extremely early stage and we refer "
    "all users to ert for real work!"
)


def _build_init_argparser(subparsers: Any) -> None:
    init_parser = subparsers.add_parser("init", help="Initialize an ERT3 workspace")
    init_parser.add_argument(
        "--example",
        help="Name of the example that would be copied "
        "to the working directory and initialised.\n"
        f"The available examples are: {', '.join(_get_ert3_example_names())}",
    )


def _build_run_argparser(subparsers: Any) -> None:
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("experiment_name", help="Name of the experiment")


def _build_export_argparser(subparsers: Any) -> None:
    export_parser = subparsers.add_parser("export", help="Export experiment")
    export_parser.add_argument("experiment_name", help="Name of the experiment")


def _build_record_argparser(subparsers: Any) -> None:
    def valid_record_file(path: Union[str, Path]) -> Path:
        path = pathlib.Path(path)
        if path.exists():
            return path
        raise argparse.ArgumentTypeError(f"No such file or directory {str(path)}")

    record_parser = subparsers.add_parser("record", help="Record operations")
    sub_record_parsers = record_parser.add_subparsers(
        dest="sub_record_cmd", help="ert3 record operations"
    )
    record_load_parser = sub_record_parsers.add_parser(
        "load", help="Load JSON records from file"
    )
    record_load_parser.add_argument("record_name", help="Name of the resulting record")
    record_load_parser.add_argument(
        "record_file",
        type=valid_record_file,
        help="Path to resource file",
    )
    sample_parser = sub_record_parsers.add_parser(
        "sample", help="Sample stochastic parameter into a record"
    )
    sample_parser.add_argument(
        "parameter_group", help="Name of the distribution group in parameters.yml"
    )
    sample_parser.add_argument("record_name", help="Name of the resulting record")
    sample_parser.add_argument(
        "ensemble_size", type=int, help="Size of ensemble of variables"
    )


def _build_status_argparser(subparsers: Any) -> None:
    subparsers.add_parser("status", help="Report the status of all experiments")


def _build_clean_argparser(subparsers: Any) -> None:
    export_parser = subparsers.add_parser("clean", help="Clean experiments")
    group = export_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "experiment_names", nargs="*", default=[], help="Name of the experiment(s)"
    )
    group.add_argument(
        "--all", action="store_true", default=False, help="Clean all experiments"
    )


def _build_argparser() -> Any:
    parser = argparse.ArgumentParser(description=_ERT3_DESCRIPTION)
    subparsers = parser.add_subparsers(dest="sub_cmd", help="ert3 commands")

    _build_init_argparser(subparsers)
    _build_run_argparser(subparsers)
    _build_export_argparser(subparsers)
    _build_record_argparser(subparsers)
    _build_status_argparser(subparsers)
    _build_clean_argparser(subparsers)

    return parser


def _get_ert3_examples_path() -> Path:
    pkg_examples_path = pathlib.Path(pkg.resource_filename("ert3_examples", ""))
    # check that examples folder exist
    if not pkg_examples_path.exists():
        raise ModuleNotFoundError(f"Examples folder {pkg_examples_path} was not found.")
    return pkg_examples_path


def _get_ert3_example_names() -> List[str]:
    pkg_examples_path = _get_ert3_examples_path()
    ert_example_names = []
    for example in pkg_examples_path.iterdir():
        if example.is_dir() and "__" not in example.name:
            ert_example_names.append(example.name)
    return ert_example_names


def _init(args: Any) -> None:
    assert args.sub_cmd == "init"

    if args.example is None:
        ert3.workspace.initialize(pathlib.Path.cwd())
    else:
        example_name = args.example
        pkg_examples_path = _get_ert3_examples_path()
        pkg_example_path = pkg_examples_path / example_name
        wd_example_path = pathlib.Path.cwd() / example_name

        # check that examples folder contains provided 'example_name'
        if not pkg_example_path.exists():
            raise ert3.exceptions.IllegalWorkspaceOperation(
                f"Example {example_name} is not a valid ert3 example.\n"
                f"Valid examples names are:  {', '.join(_get_ert3_example_names())}."
            )

        # check that we are not inside an ERT workspace already
        if ert3.workspace.load(pathlib.Path.cwd()) is not None:
            raise ert3.exceptions.IllegalWorkspaceOperation(
                "Already inside an ERT workspace."
            )

        # check that current working directory does not contain 'example_name' folder
        if not wd_example_path.is_dir():
            shutil.copytree(pkg_example_path, wd_example_path)
        else:
            raise ert3.exceptions.IllegalWorkspaceOperation(
                f"Your working directory already contains example {example_name}."
            )

        ert3.workspace.initialize(wd_example_path)


def _run(workspace: Path, args: Any) -> None:
    assert args.sub_cmd == "run"
    ert3.workspace.assert_experiment_exists(workspace, args.experiment_name)
    ensemble = _load_ensemble_config(workspace, args.experiment_name)
    stages_config = _load_stages_config(workspace)
    experiment_config = _load_experiment_config(workspace, args.experiment_name)
    ert3.engine.run(
        ensemble,
        stages_config,
        experiment_config,
        workspace,
        args.experiment_name,
    )


def _export(workspace: Path, args: Any) -> None:
    assert args.sub_cmd == "export"
    ert3.engine.export(workspace, args.experiment_name)


def _record(workspace: Path, args: Any) -> None:
    assert args.sub_cmd == "record"
    if args.sub_record_cmd == "sample":
        ert3.engine.sample_record(
            workspace, args.parameter_group, args.record_name, args.ensemble_size
        )
    elif args.sub_record_cmd == "load":
        ert3.engine.load_record(workspace, args.record_name, args.record_file)
    else:
        raise NotImplementedError(
            f"No implementation to handle record command {args.sub_record_cmd}"
        )


def _status(workspace: Path, args: Any) -> None:
    assert args.sub_cmd == "status"
    ert3.console.status(workspace)


def _clean(workspace: Path, args: Any) -> None:
    assert args.sub_cmd == "clean"
    ert3.console.clean(workspace, args.experiment_names, args.all)


def main() -> None:
    try:
        _main()
    except ert3.exceptions.ConfigValidationError as e:
        ert3.console.report_validation_errors(e)
    except ert3.exceptions.ErtError as e:
        sys.exit(e)


def _main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    # Commands that does not require an ert workspace
    if args.sub_cmd is None:
        parser.print_help()
        return
    if args.sub_cmd == "init":
        _init(args)
        return

    # Commands that does requires an ert workspace
    workspace = ert3.workspace.load(pathlib.Path.cwd())

    if workspace is None:
        raise ert3.exceptions.IllegalWorkspaceOperation("Not inside an ERT workspace.")

    if args.sub_cmd == "run":
        _run(workspace, args)
    elif args.sub_cmd == "export":
        _export(workspace, args)
    elif args.sub_cmd == "record":
        _record(workspace, args)
    elif args.sub_cmd == "status":
        _status(workspace, args)
    elif args.sub_cmd == "clean":
        _clean(workspace, args)
    else:
        raise NotImplementedError(f"No implementation to handle command {args.sub_cmd}")


def _load_ensemble_config(workspace: Path, experiment_name: str) -> EnsembleConfig:
    ensemble_config = (
        workspace / ert3.workspace.EXPERIMENTS_BASE / experiment_name / "ensemble.yml"
    )
    with open(ensemble_config) as f:
        return ert3.config.load_ensemble_config(yaml.safe_load(f))


def _load_stages_config(workspace: Path) -> StagesConfig:
    with open(workspace / "stages.yml") as f:
        sys.path.append(str(workspace))
        config = ert3.config.load_stages_config(yaml.safe_load(f))
        return config


def _load_experiment_config(workspace: Path, experiment_name: str) -> ExperimentConfig:
    experiment_config = (
        workspace / ert3.workspace.EXPERIMENTS_BASE / experiment_name / "experiment.yml"
    )
    with open(experiment_config) as f:
        return ert3.config.load_experiment_config(yaml.safe_load(f))
