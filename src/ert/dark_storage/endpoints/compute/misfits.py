from datetime import datetime
from typing import Any, Optional, Union
from uuid import UUID

import pandas as pd
from dateutil.parser import parse
from fastapi import APIRouter, Depends, status
from fastapi.responses import Response

from ert.dark_storage import exceptions as exc
from ert.dark_storage.common import (
    data_for_key,
    get_observation_keys_for_response,
    get_observations_for_obs_keys,
)
from ert.dark_storage.compute.misfits import calculate_misfits_from_pandas
from ert.dark_storage.enkf import get_storage
from ert.storage import Storage

router = APIRouter(tags=["misfits"])
DEFAULT_STORAGEREADER = Depends(get_storage)


@router.get(
    "/compute/misfits",
    responses={
        status.HTTP_200_OK: {
            "content": {"text/csv": {}},
        }
    },
)
async def get_response_misfits(
    *,
    storage: Storage = DEFAULT_STORAGEREADER,
    ensemble_id: UUID,
    response_name: str,
    realization_index: Optional[int] = None,
    summary_misfits: bool = False,
) -> Response:
    ensemble = storage.get_ensemble(ensemble_id)
    dataframe = data_for_key(ensemble, response_name)
    if realization_index is not None:
        dataframe = pd.DataFrame(dataframe.loc[realization_index]).T

    response_dict = {}
    for index, data in dataframe.iterrows():
        data_df = pd.DataFrame(data).T
        response_dict[index] = data_df

    obs_keys = get_observation_keys_for_response(ensemble, response_name)
    obs = get_observations_for_obs_keys(ensemble, obs_keys)

    if not obs_keys:
        raise ValueError(f"No observations for key {response_name}")
    if not obs:
        raise ValueError(f"Cant fetch observations for key {response_name}")
    o = obs[0]

    def parse_index(x: Any) -> Union[int, datetime]:
        try:
            return int(x)
        except ValueError:
            return parse(x)

    observation_df = pd.DataFrame(
        data={"values": o["values"], "errors": o["errors"]},
        index=[parse_index(x) for x in o["x_axis"]],
    )
    try:
        result_df = calculate_misfits_from_pandas(
            response_dict, observation_df, summary_misfits
        )
    except Exception as misfits_exc:
        raise exc.UnprocessableError(
            f"Unable to compute misfits: {misfits_exc}"
        ) from misfits_exc
    return Response(
        content=result_df.to_csv().encode(),
        media_type="text/csv",
    )
