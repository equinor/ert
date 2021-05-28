from uuid import UUID
from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm.attributes import flag_modified
from ert_storage.database import Session, get_db
from ert_storage import database_schema as ds, json_schema as js
from typing import Any, Mapping, Optional, List


router = APIRouter(tags=["experiment"])


@router.get("/experiments", response_model=List[js.ExperimentOut])
def get_experiments(
    *,
    db: Session = Depends(get_db),
) -> List[js.ExperimentOut]:
    return [
        js.ExperimentOut(
            id=exp.id,
            name=exp.name,
            ensembles=[ens.id for ens in exp.ensembles],
            priors={pri.name: pri.id for pri in exp.priors},
            metadata=exp.metadata_dict,
        )
        for exp in db.query(ds.Experiment).all()
    ]


@router.post("/experiments", response_model=js.ExperimentOut)
def post_experiments(
    *,
    db: Session = Depends(get_db),
    ens_in: js.ExperimentIn,
) -> js.ExperimentOut:
    experiment = ds.Experiment(name=ens_in.name)

    if ens_in.priors:
        db.add_all(
            ds.Prior(
                function=ds.PriorFunction.__members__[prior.function],
                experiment=experiment,
                name=name,
                argument_names=[x[0] for x in prior if isinstance(x[1], (float, int))],
                argument_values=[x[1] for x in prior if isinstance(x[1], (float, int))],
            )
            for name, prior in ens_in.priors.items()
        )

    db.add(experiment)
    db.commit()
    return js.ExperimentOut(
        id=experiment.id,
        name=experiment.name,
        ensembles=[ens.id for ens in experiment.ensembles],
        priors={pri.name: pri.id for pri in experiment.priors},
        metadata=experiment.metadata_dict,
    )


@router.get(
    "/experiments/{experiment_id}/ensembles", response_model=List[js.EnsembleOut]
)
def get_experiment_ensembles(
    *, db: Session = Depends(get_db), experiment_id: UUID
) -> List[ds.Ensemble]:
    return db.query(ds.Ensemble).join(ds.Experiment).filter_by(id=experiment_id).all()


@router.put("/experiments/{experiment_id}/metadata")
async def replace_experiment_metadata(
    *,
    db: Session = Depends(get_db),
    experiment_id: UUID,
    body: Any = Body(...),
) -> None:
    """
    Assign new metadata json
    """
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    experiment._metadata = body
    db.commit()


@router.patch("/experiments/{experiment_id}/metadata")
async def patch_experiment_metadata(
    *,
    db: Session = Depends(get_db),
    experiment_id: UUID,
    body: Any = Body(...),
) -> None:
    """
    Update metadata json
    """
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    experiment._metadata.update(body)
    flag_modified(experiment, "_metadata")
    db.commit()


@router.get("/experiments/{experiment_id}/metadata", response_model=Mapping[str, Any])
async def get_experiment_metadata(
    *,
    db: Session = Depends(get_db),
    experiment_id: UUID,
) -> Mapping[str, Any]:
    """
    Get metadata json
    """
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    return experiment.metadata_dict


@router.delete("/experiments/{experiment_id}")
def delete_experiment(*, db: Session = Depends(get_db), experiment_id: UUID) -> None:
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    db.delete(experiment)
    db.commit()
