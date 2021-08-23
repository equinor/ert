import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import ert
import ert3


def _prepare_export(
    workspace_root: Path,
    experiment_name: str,
    parameter_names: Iterable[str],
    response_names: Iterable[str],
) -> List[Dict[str, Dict[str, ert.data.Record]]]:
    data_mapping = [(pname, "input") for pname in parameter_names]
    data_mapping += [(rname, "output") for rname in response_names]

    data: List[Dict[str, Dict[str, Any]]] = []

    for record_name, data_type in data_mapping:
        ensemble_record = ert.storage.get_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=record_name,
        )

        if not data:
            data = [{"input": {}, "output": {}} for _ in ensemble_record.records]

        if isinstance(ensemble_record.records[0], ert.data.NumericalRecord):
            assert len(data) == ensemble_record.ensemble_size
            for realization, record in zip(data, ensemble_record.records):
                assert record_name not in realization[data_type]
                realization[data_type][record_name] = record.data

    return data


def export(workspace_root: Path, experiment_name: str) -> None:
    experiment_root = (
        Path(workspace_root) / ert3.workspace.EXPERIMENTS_BASE / experiment_name
    )
    ert3.workspace.assert_experiment_exists(workspace_root, experiment_name)

    if not ert3.workspace.experiment_has_run(workspace_root, experiment_name):
        raise ValueError("Cannot export experiment that has not been carried out")

    parameter_names = ert.storage.get_experiment_parameters(
        experiment_name=experiment_name
    )
    response_names = ert.storage.get_experiment_responses(
        experiment_name=experiment_name
    )

    data = _prepare_export(
        workspace_root, experiment_name, parameter_names, response_names
    )
    with open(experiment_root / "data.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
