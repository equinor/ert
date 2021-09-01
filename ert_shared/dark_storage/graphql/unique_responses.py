from datetime import datetime
from typing import Any, TYPE_CHECKING
import graphene as gr


if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo
    from ert_shared.dark_storage.graphql.ensembles import CreateEnsemble


class RecordType(gr.Enum):
    F64_MATRIX = 1
    FILE = 2


class RecordClass(gr.Enum):
    PARAMETER = 1
    RESPONSE = 2
    OTHER = 3


class UniqueResponse(gr.ObjectType):
    ensemble = gr.Field("ert_shared.dark_storage.graphql.ensembles.CreateEnsemble")
    name = gr.String(required=True)
    record_class = RecordClass(required=True)
    record_type = RecordType(required=True)
    records = gr.List("ert_shared.dark_storage.graphql.responses.Response")
    time_created = gr.DateTime()
    time_updated = gr.DateTime()

    @staticmethod
    def resolve_ensemble(root: Any, info: "ResolveInfo") -> "CreateEnsemble":
        raise NotImplementedError

    @staticmethod
    def resolve_name(root: Any, info: "ResolveInfo") -> str:
        raise NotImplementedError

    @staticmethod
    def resolve_record_class(root: Any, info: "ResolveInfo") -> RecordClass:
        raise NotImplementedError

    @staticmethod
    def resolve_record_type(root: Any, info: "ResolveInfo") -> RecordType:
        raise NotImplementedError

    @staticmethod
    def resolve_time_created(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError

    @staticmethod
    def resolve_time_updated(root: Any, info: "ResolveInfo") -> datetime:
        raise NotImplementedError
