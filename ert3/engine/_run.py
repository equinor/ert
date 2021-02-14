import ert3
from ert3.engine import _utils


def _persist_variables(
    record_name, record_source, ensemble_size, experiment_name, workspace_root
):
    if record_source[0] == "storage":
        var_name = ".".join(record_source[1:])
    elif record_source[0] == "stochastic":
        var_name = f"{experiment_name}.{record_name}"
        ert3.engine.sample_record(
            workspace_root,
            record_source[1],
            var_name,
            ensemble_size,
        )
    else:
        raise ValueError("Unknown record source location {}".format(record_source[0]))
    return var_name


def _add_record(records, record_name, record):
    if len(record) != len(records):
        raise AssertionError(
            f"Lenght of record {record_name} ({len(record)}) "
            f"does not match ensemble size ({len(records)})"
        )

    for ens_records, new_record in zip(records, record):
        if record_name in ens_records:
            raise KeyError(f"Duplicate record name {record_name}")
        ens_records[record_name] = new_record


def _load_input_records(ensemble, workspace_root, experiment_name):
    records = [{} for _ in range(ensemble.size)]
    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(".")

        var_name = _persist_variables(
            record_name, record_source, ensemble.size, experiment_name, workspace_root
        )

        input_record = ert3.storage.get_variables(workspace_root, var_name)
        _add_record(records, record_name, input_record)
    return records


def _prepare_experiment(workspace_root, experiment_name):
    if ert3.workspace.experiment_have_run(workspace_root, experiment_name):
        raise ValueError(f"Experiment {experiment_name} have been carried out.")
    ert3.storage.init_experiment(workspace_root, experiment_name)


def _prepare_evaluation(ensemble, workspace_root, experiment_name):
    input_records = _load_input_records(ensemble, workspace_root, experiment_name)
    ert3.storage.add_input_data(workspace_root, experiment_name, input_records)


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
    parameters = _load_ensemble_parameters(ensemble, workspace_root)
    input_records = ert3.algorithms.one_at_the_time(parameters)
    ert3.storage.add_input_data(workspace_root, experiment_name, input_records)


def _evaluate(ensemble, stages_config, workspace_root, experiment_name):
    input_records = ert3.storage.get_input_data(workspace_root, experiment_name)
    response = ert3.evaluator.evaluate(
        workspace_root, experiment_name, input_records, ensemble, stages_config
    )
    ert3.storage.add_output_data(workspace_root, experiment_name, response)


def run(ensemble, stages_config, experiment_config, workspace_root, experiment_name):
    _prepare_experiment(workspace_root, experiment_name)

    if experiment_config.type == "evaluation":
        _prepare_evaluation(ensemble, workspace_root, experiment_name)
    elif experiment_config.type == "sensitivity":
        _prepare_sensitivity(ensemble, workspace_root, experiment_name)
    else:
        raise ValueError(f"Unknown experiment type {experiment_config.type}")

    _evaluate(ensemble, stages_config, workspace_root, experiment_name)
