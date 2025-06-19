import json
import operator
from typing import Any
from urllib.parse import unquote
from uuid import UUID, uuid4

import polars as pl
from fastapi import APIRouter, Body, Depends, Query

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import (
    get_storage,
)
from ert.dark_storage.endpoints.responses import response_to_pandas_x_axis_fns
from ert.storage import Experiment, Storage

router = APIRouter(tags=["ensemble"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get(
    "/experiments/{experiment_id}/observations", response_model=list[js.ObservationOut]
)
def get_observations(
    *, storage: Storage = DEFAULT_STORAGE, experiment_id: UUID
) -> list[js.ObservationOut]:
    experiment = storage.get_experiment(experiment_id)
    return [
        js.ObservationOut(
            id=UUID(int=0),
            userdata={},
            errors=observation["errors"],
            values=observation["values"],
            x_axis=observation["x_axis"],
            name=observation["name"],
        )
        for observation in _get_observations(experiment)
    ]


@router.get("/ensembles/{ensemble_id}/responses/{response_key}/observations")
async def get_observations_for_response(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_key: str,
    filter_on: str | None = Query(None, description="JSON string with filters"),
) -> list[js.ObservationOut]:
    response_key = unquote(response_key)
    ensemble = storage.get_ensemble(ensemble_id)
    experiment = ensemble.experiment

    response_type = experiment.response_key_to_response_type.get(response_key, "")
    obs_keys = experiment.response_key_to_observation_key.get(response_type, {}).get(
        response_key
    )
    if not obs_keys:
        return []

    obss = _get_observations(
        ensemble.experiment,
        obs_keys,
        json.loads(filter_on) if filter_on is not None else None,
    )

    obss.sort(key=operator.itemgetter("name"))
    if not obss:
        return []

    return [
        js.ObservationOut(
            id=uuid4(),
            userdata={},
            errors=obs["errors"],
            values=obs["values"],
            x_axis=obs["x_axis"],
            name=obs["name"],
        )
        for obs in obss
    ]


def _get_observations(
    experiment: Experiment,
    observation_keys: list[str] | None = None,
    filter_on: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    observations = []

    for response_type, df in experiment.observations.items():
        if observation_keys is not None:
            df = df.filter(pl.col("observation_key").is_in(observation_keys))

        if df.is_empty():
            continue

        if filter_on is not None:
            for response_key, selected_value in filter_on.items():
                # For now we only filter on report_step
                # When we filter on more, we should infer what type to cast
                # the value to from the dtype of the polars column
                df = df.filter(pl.col(response_key).eq(int(selected_value)))

        if df.is_empty():
            continue

        x_axis_fn = response_to_pandas_x_axis_fns[response_type]
        df = df.rename(
            {
                "observation_key": "name",
                "std": "errors",
                "observations": "values",
            }
        )
        df = df.with_columns(pl.Series(name="x_axis", values=df.map_rows(x_axis_fn)))
        df = df.sort("x_axis")

        for obs_key, _obs_df in df.group_by("name"):
            observations.append(
                {
                    "name": obs_key[0],
                    "values": _obs_df["values"].to_list(),
                    "errors": _obs_df["errors"].to_list(),
                    "x_axis": _obs_df["x_axis"].to_list(),
                }
            )

    return observations
