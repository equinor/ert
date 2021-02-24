import ert3
from ert3.engine import _utils


def _prepare_experiment(workspace_root, experiment_name, ensemble):
    if ert3.workspace.experiment_have_run(workspace_root, experiment_name):
        raise ValueError(f"Experiment {experiment_name} have been carried out.")

    parameter_names = [elem.record for elem in ensemble.input]
    ert3.storage.init_experiment(
        workspace=workspace_root,
        experiment_name=experiment_name,
        parameters=parameter_names,
    )


def _prepare_experiment_record(
    record_name, record_source, ensemble_size, experiment_name, workspace_root
):
    if record_source[0] == "storage":
        assert len(record_source) == 2
        ensemble_record = ert3.storage.get_ensemble_record(
            workspace=workspace_root, record_name=record_source[1]
        )
        ert3.storage.add_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=record_name,
            ensemble_record=ensemble_record,
        )
    elif record_source[0] == "stochastic":
        ert3.engine.sample_record(
            workspace_root,
            record_source[1],
            record_name,
            ensemble_size,
            experiment_name=experiment_name,
        )
    else:
        raise ValueError("Unknown record source location {}".format(record_source[0]))


def _prepare_evaluation(ensemble, workspace_root, experiment_name):
    parameters = {}
    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(".")

        _prepare_experiment_record(
            record_name, record_source, ensemble.size, experiment_name, workspace_root
        )

        ensemble_record = ert3.storage.get_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=record_name,
        )
        parameters[record_name] = ensemble_record


def _load_ensemble_parameters(ensemble, workspace):
    parameters = _utils.load_parameters(workspace)

    ensemble_parameters = {}
    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(".")
        assert len(record_source) == 2
        assert record_source[0] == "stochastic"
        parameter_group_name = record_source[1]
        ensemble_parameters[record_name] = parameters[parameter_group_name]
    return ensemble_parameters


def _prepare_sensitivity(ensemble, workspace_root, experiment_name):
    parameter_distributions = _load_ensemble_parameters(ensemble, workspace_root)
    input_records = ert3.algorithms.one_at_the_time(parameter_distributions)

    parameters = {param.record: [] for param in ensemble.input}
    for realization in input_records:
        assert parameters.keys() == realization.keys()
        for record_name in realization:
            parameters[record_name].append(realization[record_name])

    for record_name in parameters:
        ensemble_record = ert3.data.EnsembleRecord(records=parameters[record_name])
        ert3.storage.add_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=record_name,
            ensemble_record=ensemble_record,
        )


def _store_responses(workspace_root, experiment_name, responses):
    for record_name in responses.record_names:
        ert3.storage.add_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=record_name,
            ensemble_record=responses.ensemble_records[record_name],
        )


def _load_experiment_parameters(workspace_root, experiment_name):
    parameter_names = ert3.storage.get_experiment_parameters(
        workspace=workspace_root, experiment_name=experiment_name
    )

    parameters = {}
    for parameter_name in parameter_names:
        parameters[parameter_name] = ert3.storage.get_ensemble_record(
            workspace=workspace_root,
            experiment_name=experiment_name,
            record_name=parameter_name,
        )

    return ert3.data.MultiEnsembleRecord(ensemble_records=parameters)


def _evaluate(ensemble, stages_config, workspace_root, experiment_name):
    parameters = _load_experiment_parameters(workspace_root, experiment_name)
    responses = ert3.evaluator.evaluate(
        workspace_root, experiment_name, parameters, ensemble, stages_config
    )
    _store_responses(workspace_root, experiment_name, responses)


def run(ensemble, stages_config, experiment_config, workspace_root, experiment_name):
    _prepare_experiment(workspace_root, experiment_name, ensemble)

    if experiment_config.type == "evaluation":
        _prepare_evaluation(ensemble, workspace_root, experiment_name)
    elif experiment_config.type == "sensitivity":
        _prepare_sensitivity(ensemble, workspace_root, experiment_name)
    else:
        raise ValueError(f"Unknown experiment type {experiment_config.type}")

    _evaluate(ensemble, stages_config, workspace_root, experiment_name)
