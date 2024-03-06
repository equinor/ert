import io
from itertools import chain
from typing import Any, Dict, List, Mapping, Union
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, File, Header, status
from fastapi.responses import Response
from typing_extensions import Annotated

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import (
    data_for_key,
    ensemble_parameters,
    gen_data_keys,
    get_observation_keys_for_response,
    get_observation_name,
    get_observations_for_obs_keys,
    get_all_observations,
    get_observation_for_response,
)
from ert.dark_storage.enkf import get_storage
from ert.storage import Storage

router = APIRouter(tags=["record"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)
DEFAULT_FILE = File(...)
DEFAULT_HEADER = Header("application/json")


@router.get("/ensembles/{ensemble_id}/records/{response_name}/observations")
async def get_record_observations(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_name: str,
) -> List[js.ObservationOut]:
    ensemble = storage.get_ensemble(ensemble_id)
    # obs_keys = get_observation_keys_for_response(ensemble, response_name)
    # obss = get_observations_for_obs_keys(ensemble, obs_keys)
    obs = get_observation_for_response(ensemble, response_name)
    # observations = ensemble.experiment.observations

    if obs is None:
        return []

    # ensemble.experiment.observations
    #
    # datasets = []
    # if response_name in observations:
    #    # It is not a summary
    #    datasets.append(observations[response_name])
    #
    # elif "summary" in observations and response_name in ensemble.get_summary_keyset():
    #    datasets = [ds for _, ds in observations["summary"].groupby("obs_name")]
    #    pass
    #
    # if not datasets:
    #    return []

    return [
        js.ObservationOut(
            id=uuid4(),
            userdata={},
            errors=obs.errors,
            values=obs.values,
            x_axis=obs.x_axis,
            name=obs.obs_name,
        )
    ]


@router.get(
    "/ensembles/{ensemble_id}/records/{name}",
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "text/csv": {},
                "application/x-parquet": {},
            }
        }
    },
)
async def get_ensemble_record(
    *,
    storage: Storage = DEFAULT_STORAGE,
    name: str,
    ensemble_id: UUID,
    accept: Annotated[Union[str, None], Header()] = None,
) -> Any:
    dataframe = data_for_key(storage.get_ensemble(ensemble_id), name)

    media_type = accept if accept is not None else "text/csv"
    if media_type == "application/x-parquet":
        dataframe.columns = [str(s) for s in dataframe.columns]
        stream = io.BytesIO()
        dataframe.to_parquet(stream)
        return Response(
            content=stream.getvalue(),
            media_type="application/x-parquet",
        )
    elif media_type == "application/json":
        return Response(dataframe.to_json(), media_type="application/json")
    else:
        return Response(
            content=dataframe.to_csv().encode(),
            media_type="text/csv",
        )


@router.get("/ensembles/{ensemble_id}/parameters", response_model=List[Dict[str, Any]])
async def get_ensemble_parameters(
    *, storage: Storage = DEFAULT_STORAGE, ensemble_id: UUID
) -> List[Dict[str, Any]]:
    return ensemble_parameters(storage, ensemble_id)


@router.get(
    "/ensembles/{ensemble_id}/responses", response_model=Mapping[str, js.RecordOut]
)
def get_ensemble_responses(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
) -> Mapping[str, js.RecordOut]:
    response_map: Dict[str, js.RecordOut] = {}
    ensemble = storage.get_ensemble(ensemble_id)

    response_names_with_observations = set()
    observations = ensemble.experiment.observations

    if "gen_data" in observations:
        gen_obs_ds = observations["gen_data"]
        response_names_with_observations.update(
            [
                f"{x}@{','.join(map(str, gen_obs_ds.sel(name=x).report_step.values.flatten()))}"
                for x in gen_obs_ds["name"].data
            ]
        )

        for name in gen_data_keys(ensemble):
            response_map[str(name)] = js.RecordOut(
                id=UUID(int=0),
                name=name,
                userdata={"data_origin": "GEN_DATA"},
                has_observations=name in response_names_with_observations,
            )

    if "summary" in observations:
        summary_ds = observations["summary"]
        response_names_with_observations.update(summary_ds["name"].data)

        for name in ensemble.get_summary_keyset():
            response_map[str(name)] = js.RecordOut(
                id=UUID(int=0),
                name=name,
                userdata={"data_origin": "Summary"},
                has_observations=name in response_names_with_observations,
            )

    return response_map
    # for ds in gen_obs_ds:
    #    names = ds
    #    report_step = ds.report_step.values.flatten()#

    #    # response_names_with_observations.add(response_name + "@" + str(report_step))

    return None
    # for response_type, ds in ensemble.experiment.observations.items():
    #    response_type == "summary":
    #        pass
    #    if dataset.attrs["response"] == "summary" and "name" in dataset.coords:
    #        summary_kw_names = dataset.name.values.flatten()
    #        response_names_with_observations = response_names_with_observations.union(set(summary_kw_names))
    #    else:
    #        response_name = dataset.attrs["response"]
    #        if "report_step" in dataset.coords:
    #            report_step = dataset.report_step.values.flatten()
    #        response_names_with_observations.add(response_name + "@" + str(report_step))


#
# for name in ensemble.get_summary_keyset():
#    response_map[str(name)] = js.RecordOut(
#        id=UUID(int=0),
#        name=name,
#        userdata={"data_origin": "Summary"},
#        has_observations=name in response_names_with_observations,
#    )
#
# for name in gen_data_keys(ensemble):
#    response_map[str(name)] = js.RecordOut(
#        id=UUID(int=0),
#        name=name,
#        userdata={"data_origin": "GEN_DATA"},
#        has_observations=name in response_names_with_observations,
#    )
#
# return response_map
