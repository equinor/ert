from typing import List, Iterable, Optional, TYPE_CHECKING
import graphene as gr
from graphene_sqlalchemy.utils import get_session

from ert_storage.ext.graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyMutation
from ert_storage import database_schema as ds


if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo
    from ert_storage.graphql.experiments import Experiment


class Ensemble(SQLAlchemyObjectType):
    class Meta:
        model = ds.Ensemble

    child_ensembles = gr.List(lambda: Ensemble)
    parent_ensemble = gr.Field(lambda: Ensemble)
    responses = gr.Field(
        gr.List("ert_storage.graphql.responses.Response"),
        names=gr.Argument(gr.List(gr.String), required=False, default_value=None),
    )
    unique_responses = gr.List("ert_storage.graphql.unique_responses.UniqueResponse")

    parameters = gr.List("ert_storage.graphql.parameters.Parameter")

    def resolve_child_ensembles(
        root: ds.Ensemble, info: "ResolveInfo"
    ) -> List[ds.Ensemble]:
        return [x.ensemble_result for x in root.children]

    def resolve_parent_ensemble(
        root: ds.Ensemble, info: "ResolveInfo"
    ) -> Optional[ds.Ensemble]:
        update = root.parent
        if update is not None:
            return update.ensemble_reference
        return None

    def resolve_unique_responses(
        root: ds.Ensemble, info: "ResolveInfo"
    ) -> Iterable[ds.RecordInfo]:
        session = info.context["session"]  # type: ignore
        return root.record_infos.filter_by(record_class=ds.RecordClass.response)

    def resolve_responses(
        root: ds.Ensemble, info: "ResolveInfo", names: Optional[Iterable[str]] = None
    ) -> Iterable[ds.Record]:
        session = info.context["session"]  # type: ignore
        q = (
            session.query(ds.Record)
            .join(ds.RecordInfo)
            .filter_by(ensemble=root, record_class=ds.RecordClass.response)
        )
        if names is not None:
            q = q.filter(ds.RecordInfo.name.in_(names))
        return q.all()

    def resolve_parameters(
        root: ds.Ensemble, info: "ResolveInfo"
    ) -> Iterable[ds.Record]:
        session = info.context["session"]  # type: ignore
        return (
            session.query(ds.Record)
            .join(ds.RecordInfo)
            .filter_by(ensemble=root, record_class=ds.RecordClass.parameter)
            .all()
        )


class CreateEnsemble(SQLAlchemyMutation):
    class Meta:
        model = ds.Ensemble

    class Arguments:
        parameter_names = gr.List(gr.String)
        size = gr.Int()
        active_realizations = gr.List(gr.Int)

    @staticmethod
    def mutate(
        root: Optional["Experiment"],
        info: "ResolveInfo",
        parameter_names: List[str],
        size: int,
        active_realizations: Optional[List[int]] = None,
        experiment_id: Optional[str] = None,
    ) -> ds.Ensemble:
        db = get_session(info.context)

        if experiment_id is not None:
            experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
        elif hasattr(root, "id"):
            experiment = root
        else:
            raise ValueError("ID is required")

        if active_realizations is None:
            active_realizations = list(range(size))

        ensemble = ds.Ensemble(
            parameter_names=parameter_names,
            response_names=[],
            experiment=experiment,
            size=size,
            active_realizations=active_realizations,
        )

        db.add(ensemble)
        db.commit()
        return ensemble
