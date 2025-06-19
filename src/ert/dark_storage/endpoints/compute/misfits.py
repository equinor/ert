import json
from datetime import datetime
from typing import Any
from uuid import UUID

import pandas as pd
from dateutil.parser import parse
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import Response

from ert.dark_storage import exceptions as exc
from ert.dark_storage.common import get_storage
from ert.dark_storage.compute.misfits import calculate_misfits_from_pandas
from ert.dark_storage.endpoints.observations import (
    _get_observations,
)
from ert.dark_storage.endpoints.responses import data_for_response
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
    realization_index: int | None = None,
    summary_misfits: bool = False,
    filter_on: str | None = Query(None, description="JSON string with filters"),
) -> Response:
    ensemble = storage.get_ensemble(ensemble_id)
    dataframe = data_for_response(
        ensemble,
        response_name,
        json.loads(filter_on) if filter_on is not None else None,
    )
    if realization_index is not None:
        dataframe = pd.DataFrame(dataframe.loc[realization_index]).T

    response_dict = {}
    for index, data in dataframe.iterrows():
        data_df = pd.DataFrame(data).T
        response_dict[index] = data_df

    experiment = ensemble.experiment
    response_type = experiment.response_key_to_response_type[response_name]
    obs_keys = experiment.response_key_to_observation_key[response_type].get(
        response_name, []
    )
    obs = _get_observations(
        ensemble.experiment,
        obs_keys,
        json.loads(filter_on) if filter_on is not None else None,
    )

    if not obs_keys:
        raise ValueError(f"No observations for key {response_name}")
    if not obs:
        raise ValueError(f"Cant fetch observations for key {response_name}")
    o = obs[0]

    def parse_index(x: Any) -> int | datetime:
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
