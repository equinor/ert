import json
import logging
import operator
from typing import Annotated, Any
from urllib.parse import unquote
from uuid import UUID, uuid4

import polars as pl
from fastapi import APIRouter, Body, Depends, HTTPException, Query

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import (
    get_storage,
)
from ert.dark_storage.endpoints.responses import response_to_pandas_x_axis_fns
from ert.storage import Experiment, Storage

router = APIRouter(tags=["ensemble"])
logger = logging.getLogger(__name__)


DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get(
    "/experiments/{experiment_id}/observations", response_model=list[js.ObservationOut]
)
def get_observations(
    *, storage: Storage = DEFAULT_STORAGE, experiment_id: UUID
) -> list[js.ObservationOut]:
    try:
        experiment = storage.get_experiment(experiment_id)
    except KeyError as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail="Experiment not found") from e
    except Exception as ex:
        logger.exception(ex)
        raise HTTPException(status_code=500, detail="Internal server error") from ex

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
    filter_on: Annotated[
        str | None, Query(description="JSON string with filters")
    ] = None,
) -> list[js.ObservationOut]:
    response_key = unquote(response_key)
    try:
        ensemble = storage.get_ensemble(ensemble_id)
    except KeyError as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail="Ensemble not found") from e
    except Exception as ex:
        logger.exception(ex)
        raise HTTPException(status_code=500, detail="Internal server error") from ex

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
        requested_response_type=response_type,
    )
    if not obss:
        return []

    obss.sort(key=operator.itemgetter("name"))

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
    requested_response_type: str | None = None,
) -> list[dict[str, Any]]:
    observations = []

    for stored_response_type, df in experiment.observations.items():
        if (
            requested_response_type is not None
            and stored_response_type != requested_response_type
        ):
            continue

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

        x_axis_fn = response_to_pandas_x_axis_fns[stored_response_type]
        df = df.rename(
            {
                "observation_key": "name",
                "std": "errors",
                "observations": "values",
            }
        )
        df = df.with_columns(pl.Series(name="x_axis", values=df.map_rows(x_axis_fn)))
        df = df.sort("x_axis")

        for obs_key, obs_df in df.group_by("name"):
            values = obs_df["values"].to_list()
            if all(
                "BREAKTHROUGH" in response_key
                for response_key in obs_df["response_key"].to_list()
            ):
                values = obs_df["threshold"].to_list()
            observations.append(
                {
                    "name": obs_key[0],
                    "values": values,
                    "errors": obs_df["errors"].to_list(),
                    "x_axis": obs_df["x_axis"].to_list(),
                }
            )

    return observations
