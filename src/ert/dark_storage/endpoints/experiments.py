from typing import List
from uuid import UUID

from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.enkf import get_storage
from ert.shared.storage.extraction import create_priors
from ert.storage import Storage

router = APIRouter(tags=["experiment"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get("/experiments", response_model=List[js.ExperimentOut])
def get_experiments(
    *,
    storage: Storage = DEFAULT_STORAGE,
) -> List[js.ExperimentOut]:
    return [
        js.ExperimentOut(
            id=experiment.id,
            name=experiment.name,
            ensemble_ids=[ens.id for ens in experiment.ensembles],
            priors=create_priors(experiment),
            userdata={},
        )
        for experiment in storage.experiments
    ]


@router.get("/experiments/{experiment_id}", response_model=js.ExperimentOut)
def get_experiment_by_id(
    *,
    storage: Storage = DEFAULT_STORAGE,
    experiment_id: UUID,
) -> js.ExperimentOut:
    experiment = storage.get_experiment(experiment_id)
    return js.ExperimentOut(
        name=experiment.name,
        id=experiment.id,
        ensemble_ids=[ens.id for ens in experiment.ensembles],
        priors=create_priors(experiment),
        userdata={},
    )


@router.get(
    "/experiments/{experiment_id}/ensembles", response_model=List[js.EnsembleOut]
)
def get_experiment_ensembles(
    *,
    storage: Storage = DEFAULT_STORAGE,
    experiment_id: UUID,
) -> List[js.EnsembleOut]:
    return [
        js.EnsembleOut(
            id=ensemble.id,
            experiment_id=ensemble.experiment_id,
            userdata={"name": ensemble.name},
            size=ensemble.ensemble_size,
        )
        for ensemble in storage.get_experiment(experiment_id).ensembles
    ]
