from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response

from ert_shared.dark_storage.common import data_for_key
from ert_shared.dark_storage.enkf import LibresFacade, get_res, get_name
import pandas as pd

from ert_shared.storage.extraction import create_observations
from ert_storage.compute import calculate_misfits_from_pandas

from ert_storage import exceptions as exc

router = APIRouter(tags=["misfits"])


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
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    response_name: str,
    realization_index: Optional[int] = None,
    summary_misfits: bool = False,
) -> Response:

    ensemble_name = get_name("ensemble", ensemble_id)
    dataframe = data_for_key(ensemble_name, response_name)
    if realization_index is not None:
        dataframe = pd.DataFrame(dataframe.loc[realization_index]).T

    response_dict = {}
    for index, data in dataframe.iterrows():
        data_df = pd.DataFrame(data).T
        response_dict[index] = data_df

    obs = create_observations(res)
    obs_keys = res.observation_keys(response_name)
    if not obs_keys:
        raise ValueError(f"No observations for key {response_name}")
    obs_key = obs_keys[0]
    for o in obs:
        if o["name"] == obs_key:
            observation_df = pd.DataFrame(
                data={"values": o["values"], "errors": o["errors"]},
                index=[int(x) for x in o["x_axis"]],
            )
    try:
        result_df = calculate_misfits_from_pandas(
            response_dict, observation_df, summary_misfits
        )
    except Exception as misfits_exc:
        raise exc.UnprocessableError(f"Unable to compute misfits: {misfits_exc}")
    return Response(
        content=result_df.to_csv().encode(),
        media_type="text/csv",
    )
