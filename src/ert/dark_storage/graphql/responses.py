from typing import List, Optional, TYPE_CHECKING
import graphene as gr

from ert_storage.ext.graphene_sqlalchemy import SQLAlchemyObjectType
from ert_storage import database_schema as ds


if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo


class Response(SQLAlchemyObjectType):
    class Meta:
        model = ds.Record

    name = gr.String()

    def resolve_name(root: ds.Record, info: "ResolveInfo") -> str:
        return root.name
