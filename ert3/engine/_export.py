import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any
import ert
import ert3

_SOURCE_SEPARATOR = "."


def _prepare_export_parameters(
    workspace: Path,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    stages_config: ert3.config.StagesConfig,
    ensemble_size: int,
) -> Dict[str, List[ert.data.record_data]]:
    inputs = defaultdict(list)
    step = stages_config.step_from_key(ensemble.forward_model.stage)
    if not step:
        raise ValueError(f"No step for key {ensemble.forward_model.stage}")

    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(_SOURCE_SEPARATOR, maxsplit=1)
        assert len(record_source) == 2

        if record_source[0] == "storage" or record_source[0] == "stochastic":
            exp_name = None if record_source[0] == "storage" else experiment_name
            source = record_source[1] if record_source[0] == "storage" else None
            collection = ert.storage.get_ensemble_record(
                workspace=workspace,
                record_name=record_name,
                experiment_name=exp_name,
                source=source,
                ensemble_size=ensemble_size,
            )
            # DO NOT export blob records as inputs
            if collection.record_type == ert.data.RecordType.BYTES:
                continue
            for record in collection.records:
                inputs[record_name].append(record.data)

        elif record_source[0] == "resources":
            record_mime = step.input[record_name].mime
            # DO NOT export blob records as inputs
            if record_mime == "application/octet-stream":
                continue
            file_path = workspace / "resources" / record_source[1]
            collection = ert.data.load_collection_from_file(file_path, record_mime)
            assert collection.ensemble_size == ensemble_size
            for record in collection.records:
                inputs[record_name].append(record.data)
        else:
            raise ValueError(f"Unknown record source location {record_source[0]}")

    return inputs


def _prepare_export_responses(
    workspace: Path,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    ensemble_size: int,
) -> Dict[str, List[ert.data.record_data]]:
    outputs = defaultdict(list)
    responses = [elem.record for elem in ensemble.output]
    records_url = ert.storage.get_records_url(workspace, experiment_name)

    for record_name in responses:
        for iens in range(ensemble_size):
            url = f"{records_url}/{record_name}?realization_index={iens}"
            future = ert.storage.load_record(url, ert.data.RecordType.LIST_FLOAT)
            record = asyncio.get_event_loop().run_until_complete(future)
            outputs[record_name].append(record.data)
    return outputs


def export(
    workspace_root: Path,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    stages_config: ert3.config.StagesConfig,
    ensemble_size: int,
) -> None:

    experiment_root = (
        Path(workspace_root) / ert3.workspace.EXPERIMENTS_BASE / experiment_name
    )
    ert3.workspace.assert_experiment_exists(workspace_root, experiment_name)

    if not ert3.workspace.experiment_has_run(workspace_root, experiment_name):
        raise ValueError("Cannot export experiment that has not been carried out")

    parameters = _prepare_export_parameters(
        workspace_root, experiment_name, ensemble, stages_config, ensemble_size
    )
    responses = _prepare_export_responses(
        workspace_root, experiment_name, ensemble, ensemble_size
    )
    data: List[Dict[str, Dict[str, Any]]] = []

    for iens in range(ensemble_size):
        inputs = {record: data[iens] for record, data in parameters.items()}
        outputs = {record: data[iens] for record, data in responses.items()}
        data.append({"input": inputs, "output": outputs})

    with open(experiment_root / "data.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
