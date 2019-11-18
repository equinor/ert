# -*- coding: utf-8 -*-

import configsuite

from copy import deepcopy
from collections import namedtuple

from res.enkf.jobs.scaling import job_config
from res.enkf.jobs.scaling.scaled_matrix import DataMatrix
from res.enkf.jobs.scaling.measured_data import MeasuredData
from res.enkf import LocalObsdata, ActiveList
from ecl.util.util import BoolVector
from res.enkf import RealizationStateEnum


def scaling_job(ert, user_config_dict):
    """
    Takes an instance of EnkFMain and a user config dict, will do some pre-processing on
    the user config dict, set up a ConfigSuite instance and validate the job before control
    is passed to the main job.
    """

    config_dict = _find_and_expand_wildcards(
        ert.getObservations().getMatchingKeys, user_config_dict
    )

    config = setup_configuration(config_dict, job_config.build_schema())

    if not valid_configuration(config):
        raise ValueError("Invalid configuration")
    if not valid_job(
        ert.getObservations(),
        config,
        ert.getEnsembleSize(),
        ert.getEnkfFsManager().getCurrentFileSystem(),
    ):
        raise ValueError("Invalid job")
    _observation_scaling(ert, config.snapshot)


def _observation_scaling(ert, config):
    """
    Collects data, performs scaling and applies scaling, assumes validated input.
    """
    obs = ert.getObservations()
    calculate_keys = [event.key for event in config.CALCULATE_KEYS.keys]
    index_lists = [event.index for event in config.CALCULATE_KEYS.keys]
    measured_data = MeasuredData(ert, calculate_keys, index_lists)
    measured_data.remove_nan_and_filter(
        calculate_keys, config.CALCULATE_KEYS.std_cutoff, config.CALCULATE_KEYS.alpha
    )
    matrix = DataMatrix(measured_data.data)
    matrix.std_normalization(inplace=True)

    scale_factor = matrix.get_scaling_factor(config.CALCULATE_KEYS)

    update_data = _create_active_lists(obs, config.UPDATE_KEYS.keys)

    _update_scaling(obs, scale_factor, update_data)


def _wildcard_to_dict_list(matching_keys, wildcard_key):
    """
    One of either:
    wildcard_key = {"key": "WOPT*", "index": [1,2,3]}
    wildcard_key = {"key": "WOPT*"}
    """
    if "index" in wildcard_key:
        return [{"key": key, "index": wildcard_key["index"]} for key in matching_keys]
    else:
        return [{"key": key} for key in matching_keys]


def _expand_wildcard(get_wildcard_func, wildcard_key):
    """
    Expands a wildcard
    """
    matching_keys = get_wildcard_func(wildcard_key).strings
    return _wildcard_to_dict_list(matching_keys, wildcard_key)


def _find_and_expand_wildcards(get_wildcard_func, user_dict):
    """
    Loops through the user input and identifies wildcards in observation
    names and expands them.
    """
    new_dict = deepcopy(user_dict)
    for main_key, value in user_dict.items():
        new_entries = []
        if main_key in ("UPDATE_KEYS", "CALCULATE_KEYS"):
            for val in value["keys"]:
                key = val["key"]
                if "*" in key:
                    new_entries.extend(_expand_wildcard(get_wildcard_func, key))
                else:
                    new_entries.append(val)
            new_dict[main_key]["keys"] = new_entries

    return new_dict


def setup_configuration(input_data, schema):
    """
    Creates a ConfigSuite instance and inserts default values
    """
    default_layer = job_config.get_default_values()
    config = configsuite.ConfigSuite(input_data, schema, layers=(default_layer,))
    return config


def _create_active_lists(enkf_observations, events):
    """
    Will add observation vectors to observation data. Returns
    a list of tuples mirroring the user config but also containing
    the active list where the scaling factor will be applied.
    """
    new_events = []
    observation_data = LocalObsdata("some_name", enkf_observations)
    for event in events:
        observation_data.addObsVector(enkf_observations[event.key])

        new_active_list = _get_active_list(observation_data, event.key, event.index)

        new_events.append(_make_tuple(event.key, event.index, new_active_list))

    return new_events


def _get_active_list(observation_data, key, index_list):
    """
    If the user doesn't supply an index list, the existing active
    list from the observation is used, otherwise an active list is
    created from the index list.
    """
    if index_list is not None:
        return _active_list_from_index_list(index_list)
    else:
        return observation_data.copy_active_list(key).setParent()


def _make_tuple(
    key,
    index,
    active_list,
    new_event=namedtuple("named_dict", ["key", "index", "active_list"]),
):
    return new_event(key, index, active_list)


def _update_scaling(obs, scale_factor, events):
    """
    Applies the scaling factor to the user specified index, SUMMARY_OBS needs to be treated differently
    as it only has one data point per node, compared with other observation types which have multiple
    data points per node.
    """
    for event in events:
        obs_vector = obs[event.key]
        for index, obs_node in enumerate(obs_vector):
            if obs_vector.getImplementationType().name == "SUMMARY_OBS":
                index_list = (
                    event.index if event.index is not None else range(len(obs_vector))
                )
                if index in index_list:
                    obs_node.set_std_scaling(scale_factor)
            elif obs_vector.getImplementationType().name != "SUMMARY_OBS":
                obs_node.updateStdScaling(scale_factor, event.active_list)
    print(
        "Keys: {} scaled with scaling factor: {}".format(
            [event.key for event in events], scale_factor
        )
    )


def _active_list_from_index_list(index_list):
    """
    Creates an ActiveList from a list of indexes
    :param index_list: list of index
    :type index_list:  list
    :return: Active list, a c-object with mode (ALL-ACTIVE, PARTIALLY-ACTIVE, INACTIVE) and list of indices
    :rtype: active_list
    """
    active_list = ActiveList()
    [active_list.addActiveIndex(index) for index in index_list]
    return active_list


def _set_active_lists(observation_data, key_list, active_lists):
    """
    Will make a backup of the existing active list on the observation node
    before setting the user supplied index list.
    """
    exisiting_active_lists = []
    for key, active_list in zip(key_list, active_lists):
        exisiting_active_lists.append(observation_data.copy_active_list(key))
        observation_data.setActiveList(key, active_list)
    return observation_data, exisiting_active_lists


def valid_job(observations, user_config, ensamble_size, storage):
    """
    Validates the job, assumes that the configuration is valid
    """

    calculation_keys = [entry.key for entry in user_config.snapshot.CALCULATE_KEYS.keys]
    application_keys = [entry.key for entry in user_config.snapshot.UPDATE_KEYS.keys]

    error_messages = []

    error_messages.extend(is_subset(calculation_keys, application_keys))
    obs_keys_errors = has_keys(observations, calculation_keys)
    obs_keys_present = len(obs_keys_errors) == 0
    error_messages.extend(obs_keys_errors)

    if obs_keys_present:
        error_messages.extend(
            has_data(observations, calculation_keys, ensamble_size, storage)
        )
    for error in error_messages:
        print(error)
    return len(error_messages) == 0


def has_data(observations, keys, ensamble_size, storage):
    """
    Checks that all keys have data and returns a list of error messages
    """
    error_msg = "Key: {} has no data"
    active_realizations = storage.realizationList(RealizationStateEnum.STATE_HAS_DATA)

    if len(active_realizations) == 0:
        return [error_msg.format(key) for key in keys]

    active_mask = BoolVector.createFromList(ensamble_size, active_realizations)
    return [
        error_msg.format(key)
        for key in keys
        if not observations[key].hasData(active_mask, storage)
    ]


def has_keys(observations, keys):
    """
    Checks that all keys are present in the observations and returns a list of error messages
    """
    error_msg = "Key: {} has no observations"
    return [error_msg.format(key) for key in keys if key not in observations]


def is_subset(main_list, sub_list):
    """
    Checks if all the keys in sub_list are present in main_list and returns list of error
    messages
    """
    error_msg = "Update key: {} missing from calculate keys: {}"
    missing_keys = set(sub_list).difference(set(main_list))
    return [error_msg.format(missing_key, main_list) for missing_key in missing_keys]


def valid_configuration(user_config):
    """
    Validates the configuration
    """

    for error in user_config.errors:
        print(error)
    return user_config.valid
