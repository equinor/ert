from functools import partial
import pandas as pd

import deprecation

# importlib.metadata added in python 3.8
try:
    from importlib import metadata

    __version__ = metadata.version("ert")
except ImportError:
    from pkg_resources import get_distribution

    __version__ = get_distribution("ert").version


def data_loader_factory(observation_type):
    """
    Currently, the methods returned by this factory differ. They should not.
    TODO: Remove discrepancies between returned methods.
        See https://github.com/equinor/libres/issues/808
    """
    if observation_type in ["GEN_OBS", "SUMMARY_OBS", "BLOCK_OBS"]:
        if observation_type == "GEN_OBS":
            response_loader = _load_general_response
            obs_loader = _load_general_obs
        elif observation_type == "SUMMARY_OBS":
            response_loader = _load_summary_response
            obs_loader = _load_summary_obs
        elif observation_type == "BLOCK_OBS":
            response_loader = _load_block_response
            obs_loader = _load_block_obs
        return partial(
            _extract_data,
            expected_obs=observation_type,
            response_loader=response_loader,
            obs_loader=obs_loader,
        )
    else:
        raise TypeError("Unknown observation type: {}".format(observation_type))


def _extract_data(
    facade,
    obs_keys,
    case_name,
    response_loader,
    obs_loader,
    expected_obs,
    include_data=True,
):
    if isinstance(obs_keys, str):
        obs_keys = [obs_keys]
    data_map = {}
    obs_types = [facade.get_impl_type_name_for_obs_key(key) for key in obs_keys]
    data_keys = [facade.get_data_key_for_obs_key(key) for key in obs_keys]
    if len(set(obs_types)) != 1:
        raise ObservationError(
            f"\nExpected only {expected_obs} observation type. Found: {obs_types} for {obs_keys}"
        )
    if len(set(data_keys)) != 1:
        raise ObservationError(
            f"\nExpected all obs keys ({obs_keys}) to have the same data key, found: {data_keys} "
        )
    if include_data:
        # Because all observations in this loop are pointing to the same data key,
        # we can use any of them as input to the response loader.
        data = response_loader(facade, obs_keys[0], case_name)
        data.columns = _create_multi_index(
            data.columns.to_list(), list(range(len(data.columns)))
        )
        if data.empty:
            raise ResponseError(f"No response loaded for observation keys: {obs_keys}")
    else:
        data = None
    obs = obs_loader(facade, obs_keys, case_name)
    if obs.empty:
        raise ObservationError(
            f"No observations loaded for observation keys: {obs_keys}"
        )
    for obs_key in obs_keys:
        data_for_key = _filter_df1_on_df2_by_index(data, obs[obs_key])
        data_map[obs_key] = pd.concat([obs[obs_key], data_for_key])

    return pd.concat(data_map, axis=1).astype(float)


@deprecation.deprecated(
    deprecated_in="2.19",
    removed_in="3.0",
    current_version=__version__,
    details="Use the data_loader_factory",
)
def load_general_data(facade, obs_keys, case_name, include_data=True):
    return _extract_data(
        facade,
        obs_keys,
        case_name,
        _load_general_response,
        _load_general_obs,
        "GEN_OBS",
        include_data=include_data,
    )


@deprecation.deprecated(
    deprecated_in="2.19",
    removed_in="3.0",
    current_version=__version__,
    details="Use the data_loader_factory",
)
def load_summary_data(facade, obs_keys, case_name, include_data=True):
    return _extract_data(
        facade,
        obs_keys,
        case_name,
        _load_summary_response,
        _load_summary_obs,
        "SUMMARY_OBS",
        include_data=include_data,
    )


@deprecation.deprecated(
    deprecated_in="2.19",
    removed_in="3.0",
    current_version=__version__,
    details="Use the data_loader_factory",
)
def load_block_data(facade, obs_keys, case_name, include_data=True):
    return _extract_data(
        facade,
        obs_keys,
        case_name,
        _load_block_response,
        _load_block_obs,
        "BLOCK_OBS",
        include_data=include_data,
    )


def _load_block_response(facade, obs_key, case_name):
    """
    _load_block_response is a part of the data_loader_factory, and the other
    methods returned by this factory, require case_name, so it is accepted
    here as well.
    """
    obs_vector = facade.get_observations()[obs_key]
    loader = facade.create_plot_block_data_loader(obs_vector)

    data = pd.DataFrame()
    for report_step in obs_vector.getStepList().asList():

        block_data = loader.load(facade.get_current_fs(), report_step)
        data = data.append(_get_block_measured(facade.get_ensemble_size(), block_data))

    return data


def _create_multi_index(key_index, data_index):
    arrays = [key_index, data_index]
    tuples = list(zip(*arrays))
    return pd.MultiIndex.from_tuples(tuples, names=["key_index", "data_index"])


def _load_block_obs(facade, observation_keys, case_name):
    """
    _load_block_response is a part of the data_loader_factory, and the other
    methods returned by this factory, require case_name, so it is accepted
    here as well.
    """
    observations = []
    for observation_key in observation_keys:
        obs_vector = facade.get_observations()[observation_key]
        loader = facade.create_plot_block_data_loader(obs_vector)

        data = pd.DataFrame()
        for report_step in obs_vector.getStepList().asList():
            obs_block = loader.getBlockObservation(report_step)
            index_list = [i for i in obs_block]
            index = _create_multi_index(index_list, index_list)
            data = data.append(
                pd.DataFrame(
                    [[obs_block.getValue(i) for i in obs_block]],
                    index=["OBS"],
                    columns=index,
                )
            ).append(
                pd.DataFrame(
                    [[obs_block.getStd(i) for i in obs_block]],
                    index=["STD"],
                    columns=index,
                )
            )

        data = pd.concat({observation_key: data}, axis=1)
        observations.append(data)

    return pd.concat(observations, axis=1)


def _load_general_response(facade, obs_key, case_name):
    data_key = facade.get_data_key_for_obs_key(obs_key)
    try:
        time_steps = [
            int(key.split("@")[1])
            for key in facade.all_data_type_keys()
            if facade.is_gen_data_key(key) and data_key in key
        ]
        data = pd.DataFrame()

        for time_step in time_steps:
            gen_data = facade.load_gen_data(case_name, data_key, time_step).T
            data = data.append(gen_data)
    except ValueError as err:
        raise ResponseError(
            f"No response loaded for observation key: {obs_key}"
        ) from err
    return data


def _load_general_obs(facade, observation_keys, case_name):
    observations = []
    for observation_key in observation_keys:
        obs_vector = facade.get_observations()[observation_key]
        data = []
        for time_step in obs_vector.getStepList().asList():
            # Observations and its standard deviation are a subset of the simulation data.
            # The index_list refers to indices in the simulation data. In order to
            # join these data in a DataFrame, pandas inserts the obs/std
            # data into the columns representing said indices.
            # You then get something like:
            #      observation_key
            #      0   1   2
            # OBS  NaN NaN 42
            # STD  NaN NaN 4.2
            node = obs_vector.getNode(time_step)
            index_list = [node.getIndex(nr) for nr in range(len(node))]
            index = _create_multi_index(index_list, index_list)

            data.append(
                pd.DataFrame(
                    [node.get_data_points()], columns=index, index=["OBS"]
                ).append(pd.DataFrame([node.get_std()], columns=index, index=["STD"]))
            )
        data = pd.concat(data, axis=1)
        data = pd.concat({observation_key: data}, axis=1)
        observations.append(data)

    return pd.concat(observations, axis=1)


def _get_block_measured(ensemble_size, block_data):
    data = []
    for ensemble_nr in range(ensemble_size):
        data.append(pd.DataFrame([block_data[ensemble_nr]], index=[ensemble_nr]))
    return pd.concat(data)


def _load_summary_response(facade, obs_key, case_name):
    data_key = facade.get_data_key_for_obs_key(obs_key)
    data = facade.load_all_summary_data(case_name, [data_key])
    data = data[data_key].unstack(level=-1)
    data = data.set_index(data.index.values)
    return data


def _load_summary_obs(facade, observation_keys, case_name):
    data_key = facade.get_data_key_for_obs_key(observation_keys[0])
    args = (facade, data_key, case_name)
    data = _get_summary_observations(*args)
    obs_map = {}
    for obs_key in observation_keys:
        obs_map[obs_key] = data.pipe(_remove_inactive_report_steps, *(facade, obs_key))
    return pd.concat(obs_map, axis=1)


def _get_summary_observations(facade, data_key, case_name):
    data = facade.load_observation_data(case_name, [data_key]).transpose()
    # The index from SummaryObservationCollector is {data_key} and STD_{data_key}"
    # to match the other data types this needs to be changed to OBS and STD, hence
    # the regex.
    data = data.set_index(data.index.str.replace(r"\b" + data_key, "OBS", regex=True))
    data = data.set_index(data.index.str.replace("_" + data_key, ""))
    return data


def _remove_inactive_report_steps(data, facade, observation_key, *args):
    # XXX: the data returned from the SummaryObservationCollector is not
    # specific to an observation_key, this means that the dataset contains all
    # observations on the data_key. Here the extra data is removed.
    if data.empty:
        return data

    obs_vector = facade.get_observations()[observation_key]
    active_indices = []
    for step in obs_vector.getStepList():
        active_indices.append(step - 1)
    data = data.iloc[:, active_indices]
    index = _create_multi_index(data.columns.to_list(), active_indices)
    data.columns = index
    return data


def _filter_df1_on_df2_by_index(data, obs):
    if data is None:
        return None
    else:
        return data.loc[
            :,
            data.columns.get_level_values("key_index").isin(
                obs.columns.get_level_values("key_index")
            ),
        ]


class ObservationError(Exception):
    pass


class ResponseError(Exception):
    pass
