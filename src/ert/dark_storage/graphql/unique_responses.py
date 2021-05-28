from typing import TYPE_CHECKING

from ert_storage.ext.graphene_sqlalchemy import SQLAlchemyObjectType
from ert_storage import database_schema as ds


if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo


class UniqueResponse(SQLAlchemyObjectType):
    class Meta:
        model = ds.RecordInfo
