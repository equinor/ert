import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ert.config import EnkfObservationImplementationType
from ert.config.gen_data_config import GenDataConfig
from ert.config.gen_kw_config import GenKwConfig
from ert.libres_facade import LibresFacade
from ert.storage import EnsembleReader
from ert.storage.realization_storage_state import RealizationStorageState

_logger = logging.getLogger(__name__)


def ensemble_parameter_names(res: LibresFacade) -> List[str]:
    return res.gen_kw_keys()


def ensemble_parameters(res: LibresFacade) -> List[Dict[str, Any]]:
    return [
        {"name": key, "userdata": {"data_origin": "GEN_KW"}, "labels": []}
        for key in ensemble_parameter_names(res)
    ]


def get_response_names(res: LibresFacade, ensemble: EnsembleReader) -> List[str]:
    result = ensemble.get_summary_keyset()
    result.extend(res.get_gen_data_keys().copy())
    return result





#################


def refcase_data(ensemble:EnsembleReader, key: str) -> pd.DataFrame:
    refcase = self.config.ensemble_config.refcase


    if refcase is None or key not in refcase:
        return pd.DataFrame()

    values = refcase.numpy_vector(key, report_only=False)
    dates = refcase.numpy_dates

    data = pd.DataFrame(zip(dates, values), columns=["Date", key])
    data.set_index("Date", inplace=True)

    return data.iloc[1:]



def get_history_data(
    key: str, ensemble: Optional[EnsembleReader] = None
) -> Union[pd.DataFrame, pd.Series]:
    

    if ensemble is None:
        return refcase_data(ensemble, key)

    if key not in ensemble.get_summary_keyset():
        return pd.DataFrame()

    data = gather_summary_data(ensemble, key)
    if data.empty and ensemble is not None:
        data = refcase_data(ensemble, key)

    return data



def load_all_summary_data(
    ensemble: EnsembleReader,
    keys: Optional[List[str]] = None,
    realization_index: Optional[int] = None,
) -> pd.DataFrame:
    realizations = ensemble.get_active_realizations()
    if realization_index is not None:
        if realization_index not in realizations:
            raise IndexError(f"No such realization {realization_index}")
        realizations = [realization_index]

    summary_keys = ensemble.get_summary_keyset()

    try:
        df = ensemble.load_responses("summary", tuple(realizations)).to_dataframe()
    except (ValueError, KeyError):
        return pd.DataFrame()
    df = df.unstack(level="name")
    df.columns = [col[1] for col in df.columns.values]
    df.index = df.index.rename(
        {"time": "Date", "realization": "Realization"}
    ).reorder_levels(["Realization", "Date"])
    if keys:
        summary_keys = sorted(
            [key for key in keys if key in summary_keys]
        )  # ignore keys that doesn't exist
        return df[summary_keys]
    return df


def gather_summary_data(
    ensemble: EnsembleReader,
    key: str,
    realization_index: Optional[int] = None,
) -> Union[pd.DataFrame, pd.Series]:
    data = load_all_summary_data(ensemble, [key], realization_index)
    if data.empty:
        return data
    idx = data.index.duplicated()
    if idx.any():
        data = data[~idx]
        _logger.warning(
            "The simulation data contains duplicate "
            "timestamps. A possible explanation is that your "
            "simulation timestep is less than a second."
        )
    return data.unstack(level="Realization")





def get_gen_kw_keylist(ensemble: EnsembleReader) -> List[str]:
    return [
        k
        for k, v in ensemble.experiment.parameter_info.items()
        if "_ert_kind" in v and v["_ert_kind"] == "GenKwConfig"
    ]
    # return [k for k in _get_keys(ensemble) if isinstance(k, GenKwConfig)]


def get_gen_kw_keys(ensemble: EnsembleReader) -> List[str]:
    gen_kw_keys = get_gen_kw_keylist(ensemble)

    gen_kw_list = []
    for key in gen_kw_keys:
        gen_kw_config = ensemble.experiment.parameter_configuration[key]
        assert isinstance(gen_kw_config, GenKwConfig)

        for keyword in [e.name for e in gen_kw_config.transfer_functions]:
            gen_kw_list.append(f"{key}:{keyword}")

            if gen_kw_config.shouldUseLogScale(keyword):
                gen_kw_list.append(f"LOG10_{key}:{keyword}")

    return sorted(gen_kw_list, key=lambda k: k.lower())


def _get_keys(ensemble: EnsembleReader) -> List[str]:
    return list(ensemble.experiment.parameter_configuration.keys()) + list(
        ensemble.experiment.response_configuration.keys()
    )


def get_gen_data_keylist(ensemble: EnsembleReader):
    return [
        k
        for k, v in ensemble.experiment.response_info.items()
        if "_ert_kind" in v and v["_ert_kind"] == "GenDataConfig"
    ]
    # return [k for k in _get_keys(ensemble) if isinstance(k, GenDataConfig)]


def get_gen_data_config(ensemble: EnsembleReader, key: str) -> GenDataConfig:
    config = ensemble.experiment.response_configuration[key]
    assert isinstance(config, GenDataConfig)
    return config


def get_gen_data_keys(ensemble: EnsembleReader) -> List[str]:
    # ensemble_config = self.config.ensemble_config
    # gen_data_keys = ensemble_config.get_keylist_gen_data()
    gen_data_list = []
    for key in get_gen_data_keylist(ensemble):
        # gen_data_config = ensemble_config.getNodeGenData(key)
        gen_data_config = get_gen_data_config(ensemble, key)
        if gen_data_config.report_steps is None:
            gen_data_list.append(f"{key}@0")
        else:
            for report_step in gen_data_config.report_steps:
                gen_data_list.append(f"{key}@{report_step}")
    return sorted(gen_data_list, key=lambda k: k.lower())


def load_gen_data(
    ensemble: EnsembleReader,
    key: str,
    report_step: int,
    realization_index: Optional[int] = None,
) -> pd.DataFrame:
    realizations = ensemble.realization_list(RealizationStorageState.HAS_DATA)
    if realization_index is not None:
        if realization_index not in realizations:
            raise IndexError(f"No such realization {realization_index}")
        realizations = [realization_index]
    try:
        vals = ensemble.load_responses(key, tuple(realizations)).sel(
            report_step=report_step, drop=True
        )
    except KeyError as e:
        raise KeyError(f"Missing response: {key}") from e
    index = pd.Index(vals.index.values, name="axis")
    return pd.DataFrame(
        data=vals["values"].values.reshape(len(vals.realization), -1).T,
        index=index,
        columns=realizations,
    )


def gather_gen_kw_data(
    ensemble: EnsembleReader,
    key: str,
    realization_index: Optional[int],
) -> pd.DataFrame:
    try:
        data = self.load_all_gen_kw_data(
            ensemble,
            key.split(":")[0],
            realization_index,
        )
        return data[key].to_frame().dropna()
    except KeyError:
        return pd.DataFrame()


def load_all_gen_kw_data(
    ensemble: EnsembleReader,
    group: Optional[str] = None,
    realization_index: Optional[int] = None,
) -> pd.DataFrame:
    """Loads all GEN_KW data into a DataFrame.

    This function retrieves GEN_KW data from the given ensemble reader.
    index and returns it in a pandas DataFrame.

    Args:
        ensemble: The ensemble reader from which to load the GEN_KW data.

    Returns:
        DataFrame: A pandas DataFrame containing the GEN_KW data.

    Raises:
        IndexError: If a non-existent realization index is provided.

    Note:
        Any provided keys that are not gen_kw will be ignored.
    """
    ens_mask = ensemble.get_realization_mask_from_state(
        [
            RealizationStorageState.INITIALIZED,
            RealizationStorageState.HAS_DATA,
        ]
    )
    realizations = (
        np.array([realization_index])
        if realization_index is not None
        else np.flatnonzero(ens_mask)
    )

    dataframes = []
    gen_kws = [
        config
        for config in ensemble.experiment.parameter_configuration.values()
        if isinstance(config, GenKwConfig)
    ]
    if group:
        gen_kws = [config for config in gen_kws if config.name == group]
    for key in gen_kws:
        try:
            ds = ensemble.load_parameters(
                key.name, realizations, var="transformed_values"
            )
            ds["names"] = np.char.add(f"{key.name}:", ds["names"].astype(np.str_))
            df = ds.to_dataframe().unstack(level="names")
            df.columns = df.columns.droplevel()
            for parameter in df.columns:
                if key.shouldUseLogScale(parameter.split(":")[1]):
                    df[f"LOG10_{parameter}"] = np.log10(df[parameter])
            dataframes.append(df)
        except KeyError:
            pass
    if not dataframes:
        return pd.DataFrame()

    # Format the DataFrame in a way that old code expects it
    dataframe = pd.concat(dataframes, axis=1)
    dataframe.columns.name = None
    dataframe.index.name = "Realization"

    return dataframe.sort_index(axis=1)


def data_for_key(
    ensemble: EnsembleReader,
    key: str,
    realization_index: Optional[int] = None,
) -> pd.DataFrame:
    """Returns a pandas DataFrame with the datapoints for a given key for a
    given case. The row index is the realization number, and the columns are an
    index over the indexes/dates"""
    if key.startswith("LOG10_"):
        key = key[6:]
    if key in ensemble.get_summary_keyset():
        data = load_all_summary_data(ensemble, [key], realization_index)
        data = data[key].unstack(level="Date")
    elif key in get_gen_kw_keys(ensemble):
        data = gather_gen_kw_data(ensemble, key, realization_index)
        if data.empty:
            return data
        data.columns = pd.Index([0])
    elif key in get_gen_data_keys(ensemble):
        key_parts = key.split("@")
        key = key_parts[0]
        report_step = int(key_parts[1]) if len(key_parts) > 1 else 0

        try:
            data = load_gen_data(
                ensemble,
                key,
                report_step,
                realization_index,
            ).T
        except (ValueError, KeyError):
            data = pd.DataFrame()
    else:
        return pd.DataFrame()

    try:
        return data.astype(float)
    except ValueError:
        return data


#####################


def observations_for_obs_keys(
    res: LibresFacade, obs_keys: List[str]
) -> List[Dict[str, Any]]:
    """Returns a pandas DataFrame with the datapoints for a given observation
    key for a given case. The row index is the realization number, and the
    column index is a multi-index with (obs_key, index/date, obs_index), where
    index/date is used to relate the observation to the data point it relates
    to, and obs_index is the index for the observation itself"""
    observations = []
    for key in obs_keys:
        observation = res.config.observations[key]
        obs = {
            "name": key,
            "values": list(observation.observations.values.flatten()),
            "errors": list(observation["std"].values.flatten()),
        }
        if "time" in observation.coords:
            obs["x_axis"] = _prepare_x_axis(observation.time.values.flatten())
        else:
            obs["x_axis"] = _prepare_x_axis(
                observation["index"].values.flatten(),  # type: ignore
            )

        observations.append(obs)

    return observations


def get_observation_name(res: LibresFacade, obs_keys: List[str]) -> Optional[str]:
    summary_obs = res.get_observations().getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )
    for key in obs_keys:
        observation = res.config.observations[key]
        if key in summary_obs:
            return observation.name.values.flatten()[0]
        return key
    return None


def _prepare_x_axis(
    x_axis: Sequence[Union[int, float, str, pd.Timestamp]]
) -> List[str]:
    """Converts the elements of x_axis of an observation to a string suitable
    for json. If the elements are timestamps, convert to ISO-8601 format.

    >>> _prepare_x_axis([1, 2, 3, 4])
    ['1', '2', '3', '4']
    >>> _prepare_x_axis([pd.Timestamp(x, unit="d") for x in range(3)])
    ['1970-01-01T00:00:00', '1970-01-02T00:00:00', '1970-01-03T00:00:00']
    """
    if isinstance(x_axis[0], pd.Timestamp):
        return [pd.Timestamp(x).isoformat() for x in x_axis]

    return [str(x) for x in x_axis]
