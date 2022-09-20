from typing import Any, List, Mapping
from uuid import UUID

from ert_storage import json_schema as js
from fastapi import APIRouter, Body, Depends

from ert.dark_storage.enkf import LibresFacade, get_id, get_res, get_size
from ert.shared.storage.extraction import create_priors

router = APIRouter(tags=["experiment"])


@router.get("/experiments", response_model=List[js.ExperimentOut])
def get_experiments(*, res: LibresFacade = Depends(get_res)) -> List[js.ExperimentOut]:
    priors = create_priors(res)
    return [
        js.ExperimentOut(
            name="default",
            id=get_id("experiment", "default"),
            ensemble_ids=[get_id("ensemble", case) for case in res.cases()],
            priors=priors,
            userdata={},
        )
    ]


@router.get("/experiments/{experiment_id}", response_model=js.ExperimentOut)
def get_experiment_by_id(
    *, res: LibresFacade = Depends(get_res), experiment_id: UUID
) -> js.ExperimentOut:
    return js.ExperimentOut(
        name="default",
        id=get_id("experiment", "default"),
        ensemble_ids=[get_id("ensemble", case) for case in res.cases()],
        priors=create_priors(res),
        userdata={},
    )


@router.post("/experiments", response_model=js.ExperimentOut)
def post_experiments(
    *,
    res: LibresFacade = Depends(get_res),
    ens_in: js.ExperimentIn,
) -> js.ExperimentOut:
    raise NotImplementedError


@router.get(
    "/experiments/{experiment_id}/ensembles", response_model=List[js.EnsembleOut]
)
def get_experiment_ensembles(
    *, res: LibresFacade = Depends(get_res), experiment_id: UUID
) -> List[js.EnsembleOut]:
    return [
        js.EnsembleOut(
            id=get_id("ensemble", case),
            children=[],
            parent=None,
            experiment_id=get_id("experiment", "default"),
            userdata={"name": case},
            size=get_size(res),
            parameter_names=[],
            response_names=[],
            child_ensemble_ids=[],
        )
        for case in res.cases()
    ]


@router.put("/experiments/{experiment_id}/userdata")
async def replace_experiment_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    experiment_id: UUID,
    body: Any = Body(...),
) -> None:
    raise NotImplementedError


@router.patch("/experiments/{experiment_id}/userdata")
async def patch_experiment_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    experiment_id: UUID,
    body: Any = Body(...),
) -> None:
    raise NotImplementedError


@router.get("/experiments/{experiment_id}/userdata", response_model=Mapping[str, Any])
async def get_experiment_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    experiment_id: UUID,
) -> Mapping[str, Any]:
    raise NotImplementedError


@router.delete("/experiments/{experiment_id}")
def delete_experiment(
    *, res: LibresFacade = Depends(get_res), experiment_id: UUID
) -> None:
    raise NotImplementedError
