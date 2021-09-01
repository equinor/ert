from uuid import UUID
from typing import Any, Mapping, List

from fastapi import APIRouter, Body, Depends
from ert_storage import json_schema as js

from ert_shared.dark_storage.enkf import LibresFacade, get_res


router = APIRouter(tags=["experiment"])


@router.get("/experiments", response_model=List[js.ExperimentOut])
def get_experiments(*, res: LibresFacade = Depends(get_res)) -> List[js.ExperimentOut]:
    print(res.get_current_case_name())
    raise NotImplementedError


@router.get("/experiments/{experiment_id}", response_model=js.ExperimentOut)
def get_experiment_by_id(
    *, res: LibresFacade = Depends(get_res), experiment_id: UUID
) -> js.ExperimentOut:
    raise NotImplementedError


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
    raise NotImplementedError


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
