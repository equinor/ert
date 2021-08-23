import json
import pathlib
from typing import List, Dict, Set, Union, Any
import mimetypes

import ert
import ert3

# Character used to separate record source "paths".
_SOURCE_SEPARATOR = "."


def _prepare_experiment(
    workspace_root: pathlib.Path,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    ensemble_size: int,
    parameters_config: ert3.config.ParametersConfig,
) -> None:
    if ert3.workspace.experiment_has_run(workspace_root, experiment_name):
        raise ValueError(f"Experiment {experiment_name} have been carried out.")

    parameters: Dict[str, List[str]] = {}
    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(_SOURCE_SEPARATOR, maxsplit=1)
        parameters[record_name] = _get_experiment_record_indices(
            workspace_root, record_source, parameters_config
        )
    responses = [elem.record for elem in ensemble.output]
    ert.storage.init_experiment(
        experiment_name=experiment_name,
        parameters=parameters,
        ensemble_size=ensemble_size,
        responses=responses,
    )


def _get_experiment_record_indices(
    workspace_root: pathlib.Path,
    record_source: List[str],
    parameters_config: ert3.config.ParametersConfig,
) -> List[str]:
    assert len(record_source) == 2
    source, source_record_name = record_source

    if source == "storage":
        ensemble_record = ert.storage.get_ensemble_record(
            workspace=workspace_root,
            record_name=source_record_name,
        )

        if isinstance(ensemble_record.records[0], ert.data.BlobRecord):
            return []

        indices: Set[Union[str, int]] = set()
        for record in ensemble_record.records:
            assert isinstance(record, ert.data.NumericalRecord)
            assert record.index is not None
            indices.update(record.index)
        return [str(x) for x in indices]

    elif source == "resources":
        file_path = workspace_root / "resources" / record_source[1]
        guess = mimetypes.guess_type(str(file_path))[0]
        blob_record = guess != "application/json"
        if blob_record:
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            raw_collection = json.load(f)
        return list(set().union(*(d.keys() for d in raw_collection)))

    elif source == "stochastic":
        if parameters_config[source_record_name].variables is not None:
            variables = parameters_config[source_record_name].variables
            assert variables is not None  # To make mypy checker happy
            return list(variables)
        else:
            param_size = parameters_config[source_record_name].size
            assert param_size is not None  # To make mypy checker happy
            return [str(x) for x in range(param_size)]

    raise ValueError("Unknown record source location {}".format(source))


# pylint: disable=too-many-arguments
def _prepare_experiment_record(
    record_name: str,
    record_source: List[str],
    ensemble_size: int,
    experiment_name: str,
    workspace_root: pathlib.Path,
    parameters_config: ert3.config.ParametersConfig,
    experiment_config: ert3.config.ExperimentConfig,
) -> None:
    if record_source[0] == "storage":
        assert len(record_source) == 2
        ensemble_record = ert.storage.get_ensemble_record(
            workspace=workspace_root,
            record_name=record_source[1],
        )

        # The blob record from the workspace has size one
        # We need to copy it (ensemble size) times into the experiment record
        if isinstance(ensemble_record.records[0], ert.data.BlobRecord):
            ensemble_record = ert.data.RecordCollection(
                records=[ensemble_record.records[0] for _ in range(ensemble_size)]
            )

        # Workaround to ensure compatible ensemble sizes
        # for sensitivity experiment
        if experiment_config.type == "sensitivity":
            if ensemble_record.ensemble_size != ensemble_size:
                raise ValueError(
                    f"The size of the {record_name} storage records "
                    f"does not match the size of the sensitivity records: "
                    f"is {ensemble_record.ensemble_size}, must be {ensemble_size}"
                )

        ert.storage.add_ensemble_record(
            workspace=workspace_root,
            record_name=record_name,
            ensemble_record=ensemble_record,
            experiment_name=experiment_name,
        )
    elif record_source[0] == "resources":
        file_path = workspace_root / "resources" / record_source[1]
        guess = mimetypes.guess_type(str(file_path))[0]
        blob_record = guess != "application/json"
        record_coll = ert.data.load_collection_from_file(
            file_path, blob_record, ensemble_size
        )
        ert.storage.add_ensemble_record(
            workspace=workspace_root,
            record_name=record_name,
            ensemble_record=record_coll,
            experiment_name=experiment_name,
        )

    elif record_source[0] == "stochastic":
        ert3.engine.sample_record(
            workspace_root,
            parameters_config,
            record_source[1],
            record_name,
            ensemble_size=ensemble_size,
            experiment_name=experiment_name,
        )
    else:
        raise ValueError("Unknown record source location {}".format(record_source[0]))


def _prepare_evaluation(
    ensemble: ert3.config.EnsembleConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> None:
    # This reassures mypy that the ensemble size is defined
    assert ensemble.size is not None

    _prepare_experiment(
        workspace_root, experiment_name, ensemble, ensemble.size, parameters_config
    )

    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(_SOURCE_SEPARATOR, maxsplit=1)

        _prepare_experiment_record(
            record_name,
            record_source,
            ensemble.size,
            experiment_name,
            workspace_root,
            parameters_config,
            experiment_config,
        )


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
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> None:
    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(_SOURCE_SEPARATOR, maxsplit=1)

        if record_source[0] != "stochastic":
            _prepare_experiment_record(
                record_name,
                record_source,
                ensemble_size,
                experiment_name,
                workspace_root,
                parameters_config,
                experiment_config,
            )


def _prepare_sensitivity_records(
    ensemble: ert3.config.EnsembleConfig,
    sensitivity_records: List[Dict[str, ert.data.Record]],
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> None:
    sensitivity_parameters: Dict[str, List[ert.data.Record]] = {
        param.record: []
        for param in ensemble.input
        if param.source.split(_SOURCE_SEPARATOR)[0] == "stochastic"
    }

    for realization in sensitivity_records:
        assert sensitivity_parameters.keys() == realization.keys()
        for record_name in realization:
            sensitivity_parameters[record_name].append(realization[record_name])

    for record_name in sensitivity_parameters:
        ensemble_record = ert.data.RecordCollection(
            records=sensitivity_parameters[record_name]
        )
        ert.storage.add_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=record_name,
            ensemble_record=ensemble_record,
        )


def _prepare_sensitivity(
    ensemble: ert3.config.EnsembleConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> None:
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

    ensemble_size = len(sensitivity_input_records)

    _prepare_experiment(
        workspace_root, experiment_name, ensemble, ensemble_size, parameters_config
    )

    _prepare_storage_records(
        ensemble,
        ensemble_size,
        experiment_config,
        parameters_config,
        workspace_root,
        experiment_name,
    )

    _prepare_sensitivity_records(
        ensemble,
        sensitivity_input_records,
        workspace_root,
        experiment_name,
    )


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
    model_output: ert.data.RecordCollectionMap,
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


def _store_output_records(
    workspace_root: pathlib.Path,
    experiment_name: str,
    records: ert.data.RecordCollectionMap,
) -> None:
    assert records.record_names is not None
    for record_name in records.record_names:
        ert.storage.add_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=record_name,
            ensemble_record=records.ensemble_records[record_name],
        )


def _load_experiment_parameters(
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> ert.data.RecordCollectionMap:
    parameter_names = ert.storage.get_experiment_parameters(
        experiment_name=experiment_name
    )

    parameters = {}
    for parameter_name in parameter_names:
        parameters[parameter_name] = ert.storage.get_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=parameter_name,
        )

    return ert.data.RecordCollectionMap(ensemble_records=parameters)


def _evaluate(
    ensemble: ert3.config.EnsembleConfig,
    stages_config: ert3.config.StagesConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> None:
    parameters = _load_experiment_parameters(workspace_root, experiment_name)
    output_records = ert3.evaluator.evaluate(
        workspace_root, experiment_name, parameters, ensemble, stages_config
    )
    _store_output_records(workspace_root, experiment_name, output_records)

    if experiment_config.type == "sensitivity":
        _analyze_sensitivity(
            ensemble,
            experiment_config,
            parameters_config,
            workspace_root,
            experiment_name,
            output_records,
        )


# pylint: disable=too-many-arguments
def run(
    ensemble: ert3.config.EnsembleConfig,
    stages_config: ert3.config.StagesConfig,
    experiment_config: ert3.config.ExperimentConfig,
    parameters_config: ert3.config.ParametersConfig,
    workspace_root: pathlib.Path,
    experiment_name: str,
) -> None:

    if experiment_config.type == "evaluation":
        _prepare_evaluation(
            ensemble,
            experiment_config,
            parameters_config,
            workspace_root,
            experiment_name,
        )
    elif experiment_config.type == "sensitivity":
        _prepare_sensitivity(
            ensemble,
            experiment_config,
            parameters_config,
            workspace_root,
            experiment_name,
        )
    else:
        raise ValueError(f"Unknown experiment type {experiment_config.type}")

    _evaluate(
        ensemble,
        stages_config,
        experiment_config,
        parameters_config,
        workspace_root,
        experiment_name,
    )
