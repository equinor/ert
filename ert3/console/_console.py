import argparse
import mimetypes
import os
import pathlib
import shutil
import sys
from pathlib import Path
from typing import Any, List, Union

import pkg_resources as pkg

import ert
import ert3
from ert3.workspace import Workspace
from ert3.config import DEFAULT_RECORD_MIME_TYPE
from ert_shared.async_utils import get_event_loop
from ert_shared.services import Storage

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
        "load", help="Load records from file"
    )
    record_load_parser.add_argument("record_name", help="Name of the resulting record")
    record_load_parser.add_argument(
        "record_file",
        type=valid_record_file,
        help="Path to resource file",
    )
    record_load_parser.add_argument(
        "--blob-record",
        action="store_true",
        help="Indicate that the record is a blob",
    )

    record_load_parser.add_argument(
        "--is-directory",
        action="store_true",
        default=False,
        help="Indicate that the record is a blob created as a tar from a directory",
    )

    record_load_parser.add_argument(
        "--mime-type",
        default="guess",
        help="MIME type of file. If type is 'guess', a guess is made. If a "
        + f"guess is unsuccessful, it defaults to {DEFAULT_RECORD_MIME_TYPE}. "
        + "A provided value is ignored if --blob-record is passed, as it is "
        + "then assumed that the type is 'application/octet-stream'. "
        + "Default: guess.",
        choices=tuple(("guess",) + ert.serialization.registered_types()),
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


def _build_service_argparser(subparsers: Any) -> None:
    service_parser = subparsers.add_parser("service", help="ert3 services")

    sub_service_parser = service_parser.add_subparsers(
        dest="service_cmd", help="ert3 service commands"
    )

    start_parser = sub_service_parser.add_parser("start", help="Start ert3 service")
    check_parser = sub_service_parser.add_parser("check", help="Check ert3 service")
    start_parser.add_argument(
        dest="service_name",
        choices=["storage"],
        help="ert3 storage service",
    )
    check_parser.add_argument(
        dest="service_name",
        choices=["storage"],
        help="ert3 storage service",
    )
    check_parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout for services"
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
    _build_service_argparser(subparsers)

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
        workspace = ert3.workspace.initialize(pathlib.Path.cwd())
        ert.storage.init(workspace_name=workspace.name)
    else:
        example_name = args.example
        pkg_examples_path = _get_ert3_examples_path()
        pkg_example_path = pkg_examples_path / example_name
        wd_example_path = pathlib.Path.cwd() / example_name

        # check that examples folder contains provided 'example_name'
        if not pkg_example_path.exists():
            raise ert.exceptions.IllegalWorkspaceOperation(
                f"Example {example_name} is not a valid ert3 example.\n"
                f"Valid examples names are:  {', '.join(_get_ert3_example_names())}."
            )

        # If we can create a workspace, it existed already:
        try:
            ert3.workspace.Workspace(pathlib.Path.cwd())
        except ert.exceptions.IllegalWorkspaceOperation:
            pass
        else:
            raise ert.exceptions.IllegalWorkspaceOperation(
                "Already inside an ERT workspace."
            )

        # check that current working directory does not contain 'example_name' folder
        if not wd_example_path.is_dir():
            shutil.copytree(pkg_example_path, wd_example_path)
        else:
            raise ert.exceptions.IllegalWorkspaceOperation(
                f"Your working directory already contains example {example_name}."
            )

        workspace = ert3.workspace.initialize(wd_example_path)
        ert.storage.init(workspace_name=workspace.name)


def _run(workspace: Workspace, args: Any) -> None:
    assert args.sub_cmd == "run"
    workspace.assert_experiment_exists(args.experiment_name)
    experiment_run_config = workspace.load_experiment_run_config(args.experiment_name)
    if experiment_run_config.experiment_config.type == "evaluation":
        ert3.engine.run(experiment_run_config, workspace, args.experiment_name)
    elif experiment_run_config.experiment_config.type == "sensitivity":
        ert3.engine.run_sensitivity_analysis(
            experiment_run_config, workspace, args.experiment_name
        )


def _export(workspace: Workspace, args: Any) -> None:
    assert args.sub_cmd == "export"
    experiment_run_config = workspace.load_experiment_run_config(args.experiment_name)
    ert3.engine.export(workspace, args.experiment_name, experiment_run_config)


def _record(workspace: Workspace, args: Any) -> None:
    assert args.sub_cmd == "record"
    if args.sub_record_cmd == "sample":
        parameters_config = workspace.load_parameters_config()
        collection = ert3.engine.sample_record(
            parameters_config,
            args.parameter_group,
            args.ensemble_size,
        )
        future = ert.storage.transmit_record_collection(
            record_coll=collection,
            record_name=args.record_name,
            workspace_name=workspace.name,
        )
        get_event_loop().run_until_complete(future)

    elif args.sub_record_cmd == "load":
        if args.mime_type == "guess" and not args.blob_record:
            guess = mimetypes.guess_type(str(args.record_file))[0]
            if guess:
                if ert.serialization.has_serializer(guess):
                    record_mime = guess
                else:
                    print(
                        f"Unsupported type '{guess}', defaulting to "
                        + f"'{DEFAULT_RECORD_MIME_TYPE}'."
                    )
                    record_mime = DEFAULT_RECORD_MIME_TYPE
            else:
                print(
                    f"Unable to guess what type '{args.record_file}' is, "
                    + f"defaulting to '{DEFAULT_RECORD_MIME_TYPE}'."
                )
                record_mime = DEFAULT_RECORD_MIME_TYPE
        else:
            record_mime = args.mime_type

        if args.blob_record or args.is_directory:
            record_mime = "application/octet-stream"

        get_event_loop().run_until_complete(
            ert3.engine.load_record(
                workspace,
                args.record_name,
                args.record_file,
                record_mime,
                args.is_directory,
            )
        )
    else:
        raise NotImplementedError(
            f"No implementation to handle record command {args.sub_record_cmd}"
        )


def _status(workspace: Workspace, args: Any) -> None:
    assert args.sub_cmd == "status"
    ert3.console.status(workspace)


def _clean(workspace: Workspace, args: Any) -> None:
    assert args.sub_cmd == "clean"
    ert3.console.clean(workspace, args.experiment_names, args.all)


def _service_check(args: Any) -> None:
    if args.service_name == "storage":
        try:
            Storage.connect(timeout=args.timeout)
            return  # No exception ocurred, success!
        except TimeoutError:
            sys.exit("ERROR: Ert storage not found!")

    raise SystemExit(f"{args.service_name} not implemented")


def _service_start(args: Any) -> None:
    if args.service_name == "storage":
        os.execvp("ert", ["ert", "api", "--enable-new-storage"])
    else:
        raise SystemExit(f"{args.service_name} not implemented")


def _service(args: Any) -> None:
    assert args.sub_cmd == "service"

    if args.service_cmd == "check":
        _service_check(args)
    elif args.service_cmd == "start":
        _service_start(args)
    else:
        raise SystemExit(f"{args.service_cmd} not implemented")


def main() -> None:
    try:
        _main()
    except ert.exceptions.ConfigValidationError as e:
        ert3.console.report_validation_errors(e)
    except ert.exceptions.ErtError as e:
        sys.exit(e)


def _main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    # Commands that do not require an ert workspace:
    if args.sub_cmd is None:
        parser.print_help()
        return
    if args.sub_cmd == "init":
        _init(args)
        return
    elif args.sub_cmd == "service":
        _service(args)
        return

    # The remaining commands require an existing ert workspace:
    workspace = Workspace(pathlib.Path.cwd())

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
