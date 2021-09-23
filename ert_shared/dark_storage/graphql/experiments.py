import uuid
from datetime import datetime
from uuid import UUID
from typing import Any, List, TYPE_CHECKING, Mapping

import graphene as gr

from ert_shared.dark_storage.enkf import get_res, get_id, get_size
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
        return get_id("experiment", "default")

    @staticmethod
    def resolve_time_created(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError

    @staticmethod
    def resolve_time_updated(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError

    @staticmethod
    def resolve_name(root: Any, info: "ResolveInfo") -> str:
        return root

    @staticmethod
    def resolve_userdata(root: Any, info: "ResolveInfo") -> Any:
        raise NotImplementedError


class Experiment(gr.ObjectType, _ExperimentMixin):
    ensembles = gr.List(Ensemble)
    priors = gr.JSONString()

    @staticmethod
    def resolve_ensembles(root: None, info: "ResolveInfo") -> List[Any]:
        res = get_res()
        return res.cases()

    @staticmethod
    def resolve_priors(root: Any, info: "ResolveInfo") -> Mapping:
        res = get_res()
        priors = res.gen_kw_priors()
        return priors


class CreateExperiment(gr.Mutation, _ExperimentMixin):
    class Arguments:
        name = gr.String()

    create_ensemble = CreateEnsemble.Field()
    ensembles = gr.List(CreateEnsemble)

    @staticmethod
    def mutate(root: Any, info: "ResolveInfo", name: str) -> None:
        raise NotImplementedError
