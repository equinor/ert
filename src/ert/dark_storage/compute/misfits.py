from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd


def _calculate_misfit(
    obs_value: npt.NDArray[np.float64],
    response_value: npt.NDArray[np.float64],
    obs_std: npt.NDArray[np.float64],
) -> list[float]:
    difference = response_value - obs_value
    misfit = (difference / obs_std) ** 2
    return (misfit * np.sign(difference)).tolist()


def calculate_misfits_from_pandas(
    reponses_dict: Mapping[int, pd.DataFrame],
    observation: pd.DataFrame,
    summary_misfits: bool = False,
) -> pd.DataFrame:
    """
    Compute misfits from reponses_dict (real_id, values in dataframe)
    and observation
    """
    misfits_dict = {}
    for realization_index in reponses_dict:
        misfits_dict[realization_index] = _calculate_misfit(
            observation["values"],
            reponses_dict[realization_index].loc[:, observation.index].values.flatten(),
            observation["errors"],
        )

    df = pd.DataFrame(data=misfits_dict, index=observation.index)
    if summary_misfits:
        df = pd.DataFrame([df.abs().sum(axis=0)], columns=df.columns, index=[0])
    return df.T
