import time
from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import UUID
import graphene as gr

from ert_shared.dark_storage.enkf import get_id

if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo


class Response(gr.ObjectType):
    id = gr.UUID(required=True)
    name = gr.String()
    realization_index = gr.Int()
    time_created = gr.DateTime()
    time_updated = gr.DateTime()
    userdata = gr.JSONString(required=True)

    @staticmethod
    def resolve_id(root: Any, info: "ResolveInfo") -> UUID:
        return get_id("response", root)

    @staticmethod
    def resolve_name(root: Any, info: "ResolveInfo") -> str:
        return root

    @staticmethod
    def resolve_realization_index(root: Any, info: "ResolveInfo") -> int:
        return 1

    @staticmethod
    def resolve_time_created(root: Any, info: "ResolveInfo") -> datetime:
        return datetime.now()

    @staticmethod
    def resolve_time_updated(root: Any, info: "ResolveInfo") -> datetime:
        return datetime.now()

    @staticmethod
    def resolve_userdata(root: Any, info: "ResolveInfo") -> Any:
        return {}
