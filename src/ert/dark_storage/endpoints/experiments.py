from typing import List
from uuid import UUID

from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.enkf import LibresFacade, get_res, get_storage
from ert.shared.storage.extraction import create_priors
from ert.storage import StorageReader

router = APIRouter(tags=["experiment"])

DEFAULT_LIBRESFACADE = Depends(get_res)
DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get("/experiments", response_model=List[js.ExperimentOut])
def get_experiments(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    storage: StorageReader = DEFAULT_STORAGE,
) -> List[js.ExperimentOut]:
    priors = create_priors(res)
    return [
        js.ExperimentOut(
            id=exp.id,
            name="default",
            ensemble_ids=[ens.id for ens in exp.ensembles],
            priors=priors,
            userdata={},
        )
        for exp in storage.experiments
    ]


@router.get("/experiments/{experiment_id}", response_model=js.ExperimentOut)
def get_experiment_by_id(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    storage: StorageReader = DEFAULT_STORAGE,
    experiment_id: UUID,
) -> js.ExperimentOut:
    exp = storage.get_experiment(experiment_id)

    return js.ExperimentOut(
        name="default",
        id=exp.id,
        ensemble_ids=[ens.id for ens in exp.ensembles],
        priors=create_priors(res),
        userdata={},
    )


@router.get(
    "/experiments/{experiment_id}/ensembles", response_model=List[js.EnsembleOut]
)
def get_experiment_ensembles(
    *,
    storage: StorageReader = DEFAULT_STORAGE,
    experiment_id: UUID,
) -> List[js.EnsembleOut]:
    return [
        js.EnsembleOut(
            id=ens.id,
            children=[],
            parent=None,
            experiment_id=ens.experiment_id,
            userdata={"name": ens.name},
            size=ens.ensemble_size,
            child_ensemble_ids=[],
        )
        for ens in storage.get_experiment(experiment_id).ensembles
    ]
