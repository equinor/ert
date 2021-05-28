from typing import Any, Optional
from starlette.graphql import GraphQLApp
from fastapi import APIRouter
from ert_storage.database import sessionmaker, Session
import graphene as gr
from graphql.execution.base import ExecutionResult, ResolveInfo

from ert_storage.graphql.ensembles import Ensemble, CreateEnsemble
from ert_storage.graphql.experiments import Experiment, CreateExperiment
from ert_storage import database_schema as ds


class Mutations(gr.ObjectType):
    create_experiment = CreateExperiment.Field()
    create_ensemble = CreateEnsemble.Field(experiment_id=gr.ID(required=True))


class Query(gr.ObjectType):
    experiments = gr.List(Experiment)
    experiment = gr.Field(Experiment, id=gr.ID(required=True))
    ensemble = gr.Field(Ensemble, id=gr.ID(required=True))

    @staticmethod
    def resolve_experiments(root: None, info: ResolveInfo) -> ds.Experiment:
        return Experiment.get_query(info).all()

    @staticmethod
    def resolve_experiment(root: None, info: ResolveInfo, id: str) -> ds.Experiment:
        return Experiment.get_query(info).filter_by(id=id).one()

    @staticmethod
    def resolve_ensemble(root: None, info: ResolveInfo, id: str) -> ds.Ensemble:
        return Ensemble.get_query(info).filter_by(id=id).one()


class Schema(gr.Schema):
    """
    Extended graphene Schema class, where `execute` creates a database session
    and passes it on further.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.override_session: Optional[sessionmaker] = None

    def execute(self, *args: Any, **kwargs: Any) -> ExecutionResult:
        kwargs.setdefault("context_value", {})
        if self.override_session is not None:
            session_obj = self.override_session()
        else:
            session_obj = Session()
        with session_obj as session:
            kwargs["context_value"]["session"] = session

            return super().execute(*args, **kwargs)


schema = Schema(query=Query, mutation=Mutations)
graphql_app = GraphQLApp(schema=schema)
router = APIRouter(tags=["graphql"])
router.add_route("/gql", graphql_app)
