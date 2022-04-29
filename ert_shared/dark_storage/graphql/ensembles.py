from datetime import datetime
from typing import Any, List, Optional, TYPE_CHECKING
from uuid import UUID
import graphene as gr
from ert_shared.dark_storage.common import (
    ensemble_parameters,
    ensemble_parameter_names,
    get_response_names,
    get_responses,
    get_res_from_info,
)

from ert_shared.dark_storage.enkf import (
    get_id,
    get_size,
    get_active_realizations,
)

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
    active_realizations = gr.List(gr.Int, required=False)
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
        return get_size(get_res_from_info(info))

    @staticmethod
    def resolve_active_realizations(root: Any, info: "ResolveInfo") -> List[int]:
        return get_active_realizations(get_res_from_info(info), root)

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
        return ensemble_parameter_names(get_res_from_info(info))

    @staticmethod
    def resolve_response_names(root: Any, info: "ResolveInfo") -> List[str]:
        return get_response_names(get_res_from_info(info))

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
        return get_responses(get_res_from_info(info), root)

    @staticmethod
    def resolve_unique_responses(root: Any, info: "ResolveInfo") -> "UniqueResponse":
        raise NotImplementedError

    @staticmethod
    def resolve_parameters(root: Any, info: "ResolveInfo") -> List[dict]:
        return ensemble_parameters(get_res_from_info(info))


class CreateEnsemble(gr.Mutation, _EnsembleMixin):
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
        active_realizations: List[int] = [],
        experiment_id: Optional[str] = None,
    ) -> None:
        raise NotImplementedError
