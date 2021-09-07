from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import UUID
import graphene as gr


if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo
    from ert_shared.dark_storage.graphql.unique_responses import UniqueResponse


class Parameter(gr.ObjectType):
    id = gr.UUID(required=True)
    name = gr.String()
    prior = gr.JSONString()
    realization_index = gr.Int()
    record_info = gr.Field(
        "ert_shared.dark_storage.graphql.unique_responses.UniqueResponse"
    )
    time_created = gr.DateTime()
    time_updated = gr.DateTime()
    userdata = gr.JSONString(required=True)

    @staticmethod
    def resolve_id(root: Any, info: "ResolveInfo") -> UUID:
        raise NotImplementedError

    @staticmethod
    def resolve_name(root: Any, info: "ResolveInfo") -> str:
        return root["name"]

    @staticmethod
    def resolve_prior(root: Any, info: "ResolveInfo") -> Any:
        raise NotImplementedError

    @staticmethod
    def resolve_realization_index(root: Any, info: "ResolveInfo") -> int:
        raise NotImplementedError

    @staticmethod
    def resolve_record_info(root: Any, info: "ResolveInfo") -> "UniqueResponse":
        raise NotImplementedError

    @staticmethod
    def resolve_time_created(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError

    @staticmethod
    def resolve_time_updated(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError

    @staticmethod
    def resolve_userdata(root: Any, info: "ResolveInfo") -> Any:
        raise NotImplementedError
