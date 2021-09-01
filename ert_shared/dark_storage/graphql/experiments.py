from datetime import datetime
from uuid import UUID
from typing import Any, List, TYPE_CHECKING

import graphene as gr

from ert_shared.dark_storage.graphql.ensembles import Ensemble, CreateEnsemble


if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo


class _ExperimentMixin:
    id = gr.UUID(required=True)
    time_created = gr.DateTime()
    time_updated = gr.DateTime()
    name = gr.String()
    userdata = gr.JSONString(required=True)

    @staticmethod
    def resolve_id(root: Any, info: "ResolveInfo") -> UUID:
        raise NotImplementedError

    @staticmethod
    def resolve_time_created(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError

    @staticmethod
    def resolve_time_updated(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError

    @staticmethod
    def resolve_name(root: Any, info: "ResolveInfo") -> str:
        raise NotImplementedError

    @staticmethod
    def resolve_userdata(root: Any, info: "ResolveInfo") -> Any:
        raise NotImplementedError


class Experiment(gr.ObjectType, _ExperimentMixin):
    ensembles = gr.List(Ensemble)
    priors = gr.JSONString()

    @staticmethod
    def resolve_ensembles(root: Any, info: "ResolveInfo") -> List[Any]:
        raise NotImplementedError

    @staticmethod
    def resolve_priors(root: Any, info: "ResolveInfo") -> Any:
        raise NotImplementedError


class CreateExperiment(gr.Mutation, _ExperimentMixin):
    class Arguments:
        name = gr.String()

    create_ensemble = CreateEnsemble.Field()
    ensembles = gr.List(CreateEnsemble)

    @staticmethod
    def mutate(root: Any, info: "ResolveInfo", name: str) -> None:
        raise NotImplementedError
