import time

from ert_shared import ERT
from ert_shared.storage.repository import ErtRepository
from ert_shared.storage.data_store import DataStore
from ert_data.measured import MeasuredData

from ert_shared.feature_toggling import FeatureToggling


def _create_ensemble(repository):
    facade = ERT.enkf_facade
    ensemble_name = facade.get_current_case_name()
    ensemble = repository.add_ensemble(ensemble_name)

    for i in range(facade.get_ensemble_size()):
        repository.add_realization(index=i, ensemble_name=ensemble.name)

    return ensemble


def _extract_and_dump_observations(repository, data_store):
    facade = ERT.enkf_facade

    observation_keys = [
        facade.get_observation_key(nr) for nr, _ in enumerate(facade.get_observations())
    ]

    measured_data = MeasuredData(facade, observation_keys)
    # TODO: Should save all info and info about deactivation
    measured_data.remove_inactive_observations()
    observations = measured_data.data.loc[["OBS", "STD"]]

    _dump_observations(repository=repository, data_store=data_store, observations=observations)


def _dump_observations(repository, data_store, observations):
    for key in observations.columns.get_level_values(0).unique():
        observation = observations[key]
        if repository.get_observation(name=key) is not None:
            continue
        key_indexes_df = data_store.add_data_frame(
            observation.columns.get_level_values(0).to_list()
        )
        data_indexes_df = data_store.add_data_frame(
            observation.columns.get_level_values(1).to_list()
        )
        vals_df = data_store.add_data_frame(observation.loc["OBS"].to_list())
        stds_df = data_store.add_data_frame(observation.loc["STD"].to_list())
        data_store.commit()

        repository.add_observation(
            name=key,
            key_indexes_ref=key_indexes_df.id,
            data_indexes_ref=data_indexes_df.id,
            values_ref=vals_df.id,
            stds_ref=stds_df.id,
        )


def _extract_and_dump_parameters(repository, data_store, ensemble_name):
    facade = ERT.enkf_facade

    parameter_keys = [
        key for key in facade.all_data_type_keys() if facade.is_gen_kw_key(key)
    ]
    all_parameters = {
        key: facade.gather_gen_kw_data(ensemble_name, key) for key in parameter_keys
    }

    _dump_parameters(
        repository=repository, data_store=data_store, parameters=all_parameters, ensemble_name=ensemble_name
    )


def _dump_parameters(repository, data_store, parameters, ensemble_name):
    for key, parameter in parameters.items():
        group, name = key.split(":")
        parameter_definition = repository.add_parameter_definition(
            name=name, group=group, ensemble_name=ensemble_name,
        )
        for realization_index, value in parameter.iterrows():
            value_df = data_store.add_data_frame(float(value))
            data_store.commit()

            repository.add_parameter(
                name=parameter_definition.name,
                group=parameter_definition.group,
                value_ref=value_df.id,
                realization_index=realization_index,
                ensemble_name=ensemble_name,
            )


def _extract_and_dump_responses(repository, data_store, ensemble_name):
    facade = ERT.enkf_facade

    gen_data_keys = [
        key for key in facade.all_data_type_keys() if facade.is_gen_data_key(key)
    ]
    summary_data_keys = [
        key for key in facade.all_data_type_keys() if facade.is_summary_key(key)
    ]

    gen_data_data = {
        key.split("@")[0]: facade.gather_gen_data_data(case=ensemble_name, key=key)
        for key in gen_data_keys
    }
    summary_data = {
        key: facade.gather_summary_data(case=ensemble_name, key=key)
        for key in summary_data_keys
    }

    observation_keys = repository.get_all_observation_keys()
    key_mapping = {
        facade.get_data_key_for_obs_key(key): key for key in observation_keys
    }

    _dump_response(
        repository=repository,
        data_store=data_store, 
        responses=gen_data_data,
        ensemble_name=ensemble_name,
        key_mapping=key_mapping,
    )
    _dump_response(
        repository=repository,
        data_store=data_store, 
        responses=summary_data,
        ensemble_name=ensemble_name,
        key_mapping=key_mapping,
    )


def _dump_response(repository, data_store, responses, ensemble_name, key_mapping):
    for key, response in responses.items():
        indexes_df = data_store.add_data_frame(response.index.to_list())
        data_store.commit()
        response_definition = repository.add_response_definition(
            name=key,
            indexes_ref=indexes_df.id,
            ensemble_name=ensemble_name,
            observation_name=key_mapping.get(key),
        )
        for realization_index, values in response.iteritems():
            values_df = data_store.add_data_frame(values.to_list())
            data_store.commit()
            repository.add_response(
                name=response_definition.name,
                values_ref=values_df.id,
                realization_index=realization_index,
                ensemble_name=ensemble_name,
            )


def dump_to_new_storage(repository=None, data_store=None):
    if not FeatureToggling.is_enabled("new-storage"):
        return

    start_time = time.time()
    print("Starting extraction...")

    # TODO: What to do if the ensemble already exist in the database?

    if repository is None:
        repository = ErtRepository()
    
    if data_store is None:
        data_store = DataStore()

    with repository:
        ensemble = _create_ensemble(repository)
        _extract_and_dump_observations(repository=repository, data_store=data_store)
        _extract_and_dump_parameters(repository=repository, data_store=data_store, ensemble_name=ensemble.name)
        _extract_and_dump_responses(repository=repository, data_store=data_store, ensemble_name=ensemble.name)
        repository.commit()

    end_time = time.time()
    print("Extraction done... (Took {:.2f} seconds)".format(end_time - start_time))
    print(
        "All ensembles in database: {}".format(
            ", ".join([ensemble.name for ensemble in repository.get_all_ensembles()])
        )
    )
