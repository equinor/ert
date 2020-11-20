import ert3

import argparse
import json
from pathlib import Path
import random
import sys
import ert3


def _locate_ert_workspace_root(path):
    path = Path(path)
    while True:
        if (path / ert3._WORKSPACE_DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


_ERT3_DESCRIPTION = (
    "ert3 is an ensemble-based tool for uncertainty studies.\n"
    "\nWARNING: the tool is currently in an extremely early stage and we refer "
    "all users to ert for real work!"
)


def _build_argparser():
    parser = argparse.ArgumentParser(description=_ERT3_DESCRIPTION)
    subparsers = parser.add_subparsers(dest="sub_cmd", help="ert3 commands")

    subparsers.add_parser("init", help="Initialize an ERT3 workspace")

    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("experiment_name", help="Name of the experiment")

    export_parser = subparsers.add_parser("export", help="Export experiment")
    export_parser.add_argument("experiment_name", help="Name of the experiment")

    return parser


def _init_workspace(path):
    path = Path(path)
    if _locate_ert_workspace_root(path) is not None:
        sys.exit("Already inside an ERT workspace")

    ert3.storage.Storage.init_storage(path)


def _assert_experiment(workspace_root, experiment_name):
    experiment_root = Path(workspace_root) / experiment_name
    if not experiment_root.is_dir():
        raise ValueError(
            f"{experiment_name} is not an experiment "
            f"within the workspace {workspace_root}"
        )


def _experiment_have_run(workspace_root, experiment_name):
    experiments = ert3.storage.get_experiment_names(workspace_root)
    return experiment_name in experiments


def _generate_coefficients():
    return [
        {
            "coefficients": {
                "a": random.gauss(0, 1),
                "b": random.gauss(0, 1),
                "c": random.gauss(0, 1),
            }
        }
        for _ in range(1000)
    ]


def _run_experiment(workspace_root, experiment_name):
    _assert_experiment(workspace_root, experiment_name)

    if _experiment_have_run(workspace_root, experiment_name):
        raise ValueError(f"Experiment {experiment_name} have been carried out.")

    ert3.storage.init_experiment(workspace_root, experiment_name)

    coefficients = _generate_coefficients()
    with ert3.storage.Storage(workspace_root, experiment_name) as storage:
        storage.input_data = coefficients
        storage.output_data = ert3.evaluator.evaluate(coefficients)


def _export(workspace_root, experiment_name):
    experiment_root = Path(workspace_root) / experiment_name
    _assert_experiment(workspace_root, experiment_name)

    if not _experiment_have_run(workspace_root, experiment_name):
        raise ValueError("Cannot export experiment that has not been carried out")
    with ert3.storage.Storage(workspace_root, experiment_name) as storage:
        input_data = storage.input_data
        output_data = storage.output_data
    with open(experiment_root / "data.json", "w") as f:
        json.dump(_reformat_input_output(input_data, output_data), f)


def _reformat_input_output(input_data, output_data):
    return [
        {"input": _input_data, "output": _output_data}
        for _input_data, _output_data in zip(input_data, output_data)
    ]


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    # Commands that does not require an ert workspace
    if args.sub_cmd is None:
        parser.print_help()
        return
    if args.sub_cmd == "init":
        _init_workspace(Path.cwd())
        return

    # Commands that does requires an ert workspace
    workspace = _locate_ert_workspace_root(Path.cwd())
    if workspace is None:
        sys.exit("Not inside an ERT workspace")
    if args.sub_cmd == "run":
        _run_experiment(workspace, args.experiment_name)
        return
    if args.sub_cmd == "export":
        _export(workspace, args.experiment_name)
        return
