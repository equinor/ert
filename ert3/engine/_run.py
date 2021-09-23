import asyncio
import json
import pathlib
from typing import List, Dict, Any

import ert
import ert3

# Character used to separate record source "paths".
_SOURCE_SEPARATOR = "."


def _prepare_experiment(
    workspace_root: pathlib.Path,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    ensemble_size: int,
) -> None:
    if ert3.workspace.experiment_has_run(workspace_root, experiment_name):
        raise ValueError(f"Experiment {experiment_name} have been carried out.")

    parameters = [elem.record for elem in ensemble.input]
    responses = [elem.record for elem in ensemble.output]
    ert.storage.init_experiment(
        experiment_name=experiment_name,
        parameters=parameters,
        ensemble_size=ensemble_size,
        responses=responses,
    )


# pylint: disable=too-many-arguments
def _prepare_experiment_record(
    record_name: str,
    record_source: List[str],
    record_mime: str,
    ensemble_size: int,
    experiment_name: str,
    workspace_root: pathlib.Path,
    parameters_config: ert3.config.ParametersConfig,
    experiment_config: ert3.config.ExperimentConfig,
) -> Dict[int, Dict[str, ert.storage.StorageRecordTransmitter]]:
    assert len(record_source) == 2
    if record_source[0] == "storage":
        records_url = ert.storage.get_records_url(workspace_root)
        future = ert.storage.get_record_storage_transmitters(
            records_url=records_url,
            record_name=record_name,
            record_source=record_source[1],
            ensemble_size=ensemble_size,
        )
        return asyncio.get_event_loop().run_until_complete(future)

    elif record_source[0] == "resources":
        file_path = workspace_root / "resources" / record_source[1]
        collection = ert.data.load_collection_from_file(file_path, record_mime)
        future = ert.storage.transmit_record_collection(
            record_coll=collection,
            record_name=record_name,
            workspace=workspace_root,
            experiment_name=experiment_name,
        )
        transmitters = asyncio.get_event_loop().run_until_complete(future)
        return transmitters

    elif experiment_config.type != "sensitivity" and record_source[0] == "stochastic":
        collection = ert3.engine.sample_record(
            parameters_config,
            record_source[1],
            ensemble_size=ensemble_size,
        )
        future = ert.storage.transmit_record_collection(
            record_coll=collection,
            record_name=record_name,
            workspace=workspace_root,
            experiment_name=experiment_name,
        )
        transmitters = asyncio.get_event_loop().run_until_complete(future)
        return transmitters
    elif experiment_config.type == "sensitivity" and record_source[0] == "stochastic":
        return {}
    else:
        raise ValueError(f"Unknown record source location {record_source[0]}")


def _load_sensitivity_parameters(
    ensemble: ert3.config.EnsembleConfig,
    parameters_config: ert3.config.ParametersConfig,
) -> Dict[str, ert3.stats.Distribution]:
    all_distributions = {
        param.name: param.as_distribution() for param in parameters_config
    }

    sensitivity_parameters = {}
    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(_SOURCE_SEPARATOR)
        if record_source[0] == "stochastic":
            assert len(record_source) == 2
            group_name = record_source[1]
            sensitivity_parameters[record_name] = all_distributions[group_name]
    return sensitivity_parameters


def _prepare_storage_records(
    ensemble: ert3.config.EnsembleConfig,
    ensemble_size: int,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
    stages_config: ert3.config.StagesConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    step = stages_config.step_from_key(ensemble.forward_model.stage)
    if not step:
        raise ValueError(f"No step for key {ensemble.forward_model.stage}")

    transmitter_map: Dict[int, Dict[str, ert.data.RecordTransmitter]] = {
        iens: {} for iens in range(ensemble_size)
    }
    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(_SOURCE_SEPARATOR, maxsplit=1)
        record_mime = step.input[record_name].mime
        transmitters = _prepare_experiment_record(
            record_name,
            record_source,
            record_mime,
            ensemble_size,
            experiment_name,
            workspace_root,
            parameters_config,
            experiment_config,
        )

        for iens, trans_map in transmitters.items():
            transmitter_map[iens].update(trans_map)
    return transmitter_map


def _prepare_sensitivity_records(
    ensemble: ert3.config.EnsembleConfig,
    sensitivity_records: List[Dict[str, ert.data.Record]],
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    sensitivity_parameters: Dict[str, List[ert.data.Record]] = {
        param.record: []
        for param in ensemble.input
        if param.source.split(_SOURCE_SEPARATOR)[0] == "stochastic"
    }

    for realization in sensitivity_records:
        assert sensitivity_parameters.keys() == realization.keys()
        for record_name in realization:
            sensitivity_parameters[record_name].append(realization[record_name])

    transmitter_map: Dict[int, Dict[str, ert.data.RecordTransmitter]] = {
        iens: {} for iens in range(len(sensitivity_records))
    }
    for record_name in sensitivity_parameters:
        ensemble_record = ert.data.RecordCollection(
            records=sensitivity_parameters[record_name]
        )
        future = ert.storage.transmit_record_collection(
            record_coll=ensemble_record,
            record_name=record_name,
            workspace=workspace_root,
            experiment_name=experiment_name,
        )
        transmitters = asyncio.get_event_loop().run_until_complete(future)
        for iens, trans_map in transmitters.items():
            transmitter_map[iens].update(trans_map)
    return transmitter_map


def _prepare_sensitivity(
    ensemble: ert3.config.EnsembleConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
) -> List[Dict[str, ert.data.Record]]:
    sensitivity_distributions = _load_sensitivity_parameters(
        ensemble, parameters_config
    )

    if experiment_config.algorithm == "one-at-a-time":
        sensitivity_input_records = ert3.algorithms.one_at_the_time(
            sensitivity_distributions, tail=experiment_config.tail
        )
    elif experiment_config.algorithm == "fast":
        sensitivity_input_records = ert3.algorithms.fast_sample(
            sensitivity_distributions,
            experiment_config.harmonics,
            experiment_config.sample_size,
        )
    else:
        raise ValueError(f"Unknown algorithm {experiment_config.algorithm}")
    return sensitivity_input_records


def _store_sensitivity_analysis(
    analysis: Dict[Any, Any],
    workspace_root: pathlib.Path,
    experiment_name: str,
    output_file: str,
) -> None:
    experiment_root = (
        pathlib.Path(workspace_root) / ert3.workspace.EXPERIMENTS_BASE / experiment_name
    )
    with open(experiment_root / output_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f)


def _analyze_sensitivity(
    ensemble: ert3.config.EnsembleConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
    model_output: Dict[int, Dict[str, ert.data.RecordTransmitter]],
) -> None:
    if experiment_config.algorithm == "one-at-a-time":
        # There is no post analysis step for the one-at-a-time algorithm
        pass
    elif experiment_config.algorithm == "fast":
        sensitivity_parameters = _load_sensitivity_parameters(
            ensemble, parameters_config
        )
        analysis = ert3.algorithms.fast_analyze(
            sensitivity_parameters, model_output, experiment_config.harmonics
        )
        _store_sensitivity_analysis(
            analysis, workspace_root, experiment_name, "fast_analysis.json"
        )
    else:
        raise ValueError(
            "Unable to determine analysis step "
            f"for algorithm {experiment_config.algorithm}"
        )


def _evaluate(
    parameters: Dict[int, Dict[str, ert.data.RecordTransmitter]],
    ensemble: ert3.config.EnsembleConfig,
    stages_config: ert3.config.StagesConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    if ensemble.storage_type == "ert_storage":
        storage_path = ert.storage.get_records_url(workspace_root, experiment_name)
    else:
        evaluation_tmp_dir = (
            pathlib.Path(workspace_root)
            / ert3._WORKSPACE_DATA_ROOT
            / "tmp"
            / experiment_name
        )
        storage_path = str(evaluation_tmp_dir / ".my_storage")

    return ert3.evaluator.evaluate(storage_path, parameters, ensemble, stages_config)


# pylint: disable=too-many-arguments
def run(
    ensemble: ert3.config.EnsembleConfig,
    stages_config: ert3.config.StagesConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> None:
    # This reassures mypy that the ensemble size is defined
    assert ensemble.size is not None
    ensemble_size = ensemble.size

    _prepare_experiment(workspace_root, experiment_name, ensemble, ensemble_size)

    parameters = _prepare_storage_records(
        ensemble,
        ensemble_size,
        experiment_config,
        parameters_config,
        stages_config,
        workspace_root,
        experiment_name,
    )

    _evaluate(
        parameters,
        ensemble,
        stages_config,
        workspace_root,
        experiment_name,
    )


# pylint: disable=too-many-arguments
def run_sensitivity_analysis(
    ensemble: ert3.config.EnsembleConfig,
    stages_config: ert3.config.StagesConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> None:
    sensitivity_input_records = _prepare_sensitivity(
        ensemble,
        experiment_config,
        parameters_config,
    )
    ensemble_size = len(sensitivity_input_records)

    _prepare_experiment(workspace_root, experiment_name, ensemble, ensemble_size)

    parameters = _prepare_sensitivity_records(
        ensemble,
        sensitivity_input_records,
        workspace_root,
        experiment_name,
    )

    storage_transmitters = _prepare_storage_records(
        ensemble,
        ensemble_size,
        experiment_config,
        parameters_config,
        stages_config,
        workspace_root,
        experiment_name,
    )
    for iens, trans_map in storage_transmitters.items():
        parameters[iens].update(trans_map)

    output_transmitters = _evaluate(
        parameters,
        ensemble,
        stages_config,
        workspace_root,
        experiment_name,
    )
    _analyze_sensitivity(
        ensemble,
        experiment_config,
        parameters_config,
        workspace_root,
        experiment_name,
        output_transmitters,
    )


def get_ensemble_size(
    ensemble: ert3.config.EnsembleConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
) -> int:
    if experiment_config.type == "sensitivity":
        return len(_prepare_sensitivity(ensemble, experiment_config, parameters_config))
    else:
        assert ensemble.size is not None
        return ensemble.size
