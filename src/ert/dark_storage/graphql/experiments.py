from typing import List, Mapping, TYPE_CHECKING
import graphene as gr
from graphene_sqlalchemy.utils import get_session

from ert_storage.ext.graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyMutation

from ert_storage.graphql.ensembles import Ensemble, CreateEnsemble
from ert_storage.graphql.updates import Update
from ert_storage import database_schema as ds, json_schema as js
from ert_storage.endpoints.experiments import experiment_priors_to_dict


if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo


class Experiment(SQLAlchemyObjectType):
    class Meta:
        model = ds.Experiment

    ensembles = gr.List(Ensemble)
    priors = gr.JSONString()

    def resolve_ensembles(
        root: ds.Experiment, info: "ResolveInfo"
    ) -> List[ds.Ensemble]:
        return root.ensembles

    def resolve_priors(root: ds.Experiment, info: "ResolveInfo") -> Mapping[str, dict]:
        return experiment_priors_to_dict(root)


class CreateExperiment(SQLAlchemyMutation):
    class Arguments:
        name = gr.String()

    class Meta:
        model = ds.Experiment

    create_ensemble = CreateEnsemble.Field()

    @staticmethod
    def mutate(root: None, info: "ResolveInfo", name: str) -> ds.Experiment:
        db = get_session(info.context)

        experiment = ds.Experiment(name=name)

        db.add(experiment)
        db.commit()

        return experiment
