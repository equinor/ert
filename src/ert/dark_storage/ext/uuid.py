from uuid import UUID as SystemUUID
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator, CHAR

import graphene
from graphene import UUID as GrapheneUUID
from graphene_sqlalchemy.converter import convert_sqlalchemy_type
from graphene_sqlalchemy.registry import Registry


class UUID(TypeDecorator):
    """Platform-independent UUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgresUUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return str(value)
        else:
            if not isinstance(value, SystemUUID):
                return "%.32x" % SystemUUID(value).int
            else:
                return "%.32x" % value.int

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return value
        else:
            if not isinstance(value, SystemUUID):
                value = SystemUUID(value)
            return value


@convert_sqlalchemy_type.register(UUID)
def convert_column_to_uuid(
    type: UUID, column: sa.Column, registry: Optional[Registry] = None
) -> graphene.types.structures.Structure:
    return GrapheneUUID
