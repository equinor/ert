from datetime import datetime
from ert_shared.libres_facade import LibresFacade
from typing import Any, List, Optional, TYPE_CHECKING
from uuid import UUID
from fastapi.param_functions import Depends
import graphene as gr
from graphene.types.scalars import ID
from ert_shared.dark_storage.common import ensemble_parameters, get_response_names

from ert_shared.dark_storage.enkf import get_id, get_res, get_size

if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo
    from ert_shared.dark_storage.graphql.experiments import Experiment
    from ert_shared.dark_storage.graphql.updates import Update
    from ert_shared.dark_storage.graphql.parameters import Parameter
    from ert_shared.dark_storage.graphql.unique_responses import UniqueResponse
    from ert_shared.dark_storage.graphql.responses import Response


class _EnsembleMixin:
    id = gr.UUID(required=True)
    size = gr.Int(required=True)
    time_created = gr.DateTime()
    time_updated = gr.DateTime()
    experiment = gr.Field(
        "ert_shared.dark_storage.graphql.experiments.CreateExperiment"
    )
    parameter_names = gr.List(gr.String, required=True)
    response_names = gr.List(gr.String, required=True)
    userdata = gr.JSONString(required=True)

    children = gr.List("ert_shared.dark_storage.graphql.updates.Update")
    parent = gr.Field("ert_shared.dark_storage.graphql.updates.Update")

    @staticmethod
    def resolve_id(root: Any, info: "ResolveInfo") -> UUID:
        return get_id("ensemble", root)

    @staticmethod
    def resolve_size(root: Any, info: "ResolveInfo") -> int:
        return get_size()

    @staticmethod
    def resolve_time_created(root: Any, info: "ResolveInfo") -> datetime:
        return datetime.now()

    @staticmethod
    def resolve_time_updated(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError

    @staticmethod
    def resolve_time_experiment(root: Any, info: "ResolveInfo") -> "Experiment":
        raise NotImplementedError

    @staticmethod
    def resolve_parameter_names(root: Any, info: "ResolveInfo") -> List[str]:
        return [parameter["name"] for parameter in ensemble_parameters(root)]

    @staticmethod
    def resolve_response_names(root: Any, info: "ResolveInfo") -> List[str]:
        return get_response_names()

    @staticmethod
    def resolve_userdata(root: Any, info: "ResolveInfo") -> Any:
        return {"name": root}

    @staticmethod
    def resolve_children(root: Any, info: "ResolveInfo") -> List["Update"]:
        return []

    @staticmethod
    def resolve_parent(root: Any, info: "ResolveInfo") -> "Update":
        return None

    @staticmethod
    def resolve_experiment(root: Any, info: "ResolveInfo") -> "Experiment":
        return "default"


class Ensemble(gr.ObjectType, _EnsembleMixin):
    child_ensembles = gr.List(lambda: Ensemble)
    parent_ensemble = gr.Field(lambda: Ensemble)
    responses = gr.Field(
        gr.List("ert_shared.dark_storage.graphql.responses.Response"),
        names=gr.Argument(gr.List(gr.String), required=False, default_value=None),
    )
    unique_responses = gr.List(
        "ert_shared.dark_storage.graphql.unique_responses.UniqueResponse"
    )

    parameters = gr.List("ert_shared.dark_storage.graphql.parameters.Parameter")

    @staticmethod
    def resolve_child_ensembles(root: Any, info: "ResolveInfo") -> "Ensemble":
        return []

    @staticmethod
    def resolve_parent_ensemble(root: Any, info: "ResolveInfo") -> "Ensemble":
        return None

    @staticmethod
    def resolve_responses(root: Any, info: "ResolveInfo") -> "Response":
        return _EnsembleMixin.resolve_response_names(root, info)

    @staticmethod
    def resolve_unique_responses(root: Any, info: "ResolveInfo") -> "UniqueResponse":
        raise NotImplementedError

    @staticmethod
    def resolve_parameters(root: Any, info: "ResolveInfo") -> List[dict]:
        return ensemble_parameters(root)


class CreateEnsemble(gr.Mutation, _EnsembleMixin):
    class Arguments:
        parameter_names = gr.List(gr.String)
        size = gr.Int()

    @staticmethod
    def mutate(
        root: Optional["Experiment"],
        info: "ResolveInfo",
        parameter_names: List[str],
        size: int,
        experiment_id: Optional[str] = None,
    ) -> None:
        raise NotImplementedError
