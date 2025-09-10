from __future__ import annotations

from datetime import datetime

from ._observation_declaration import (
    ConfContent,
    GenObsValues,
    HistoryValues,
    SummaryValues,
)
from .ensemble_config import EnsembleConfig
from .observation_vector import ObsVector
from .observations import EnkfObs
from .parsing import (
    ErrorInfo,
    HistorySource,
    ObservationConfigError,
)


def create_observations(
    obs_config_content: ConfContent,
    ensemble_config: EnsembleConfig,
    time_map: list[datetime] | None,
    history: HistorySource,
) -> EnkfObs:
    if not obs_config_content:
        return EnkfObs({}, [])
    obs_vectors: dict[str, ObsVector] = {}
    obs_time_list: list[datetime] = []
    if ensemble_config.refcase is not None:
        obs_time_list = ensemble_config.refcase.all_dates
    elif time_map is not None:
        obs_time_list = time_map

    time_len = len(obs_time_list)
    config_errors: list[ErrorInfo] = []
    for obs_name, values in obs_config_content:
        try:
            if type(values) is HistoryValues:
                obs_vectors.update(
                    **EnkfObs._handle_history_observation(
                        ensemble_config,
                        values,
                        obs_name,
                        history,
                        time_len,
                    )
                )
            elif type(values) is SummaryValues:
                obs_vectors.update(
                    **EnkfObs._handle_summary_observation(
                        values,
                        obs_name,
                        obs_time_list,
                        bool(ensemble_config.refcase),
                    )
                )
            elif type(values) is GenObsValues:
                obs_vectors.update(
                    **EnkfObs._handle_general_observation(
                        ensemble_config,
                        values,
                        obs_name,
                        obs_time_list,
                        bool(ensemble_config.refcase),
                    )
                )
            else:
                config_errors.append(
                    ErrorInfo(
                        message=(
                            f"Unknown ObservationType {type(values)} for {obs_name}"
                        )
                    ).set_context(obs_name)
                )
                continue
        except ObservationConfigError as err:
            config_errors.extend(err.errors)
        except ValueError as err:
            config_errors.append(ErrorInfo(message=str(err)).set_context(obs_name))

    if config_errors:
        raise ObservationConfigError.from_collected(config_errors)

    return EnkfObs(obs_vectors, obs_time_list)
