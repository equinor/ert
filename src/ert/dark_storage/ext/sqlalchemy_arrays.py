"""
This module adds the FloatArray, StringArray and IntArray column types. In Postgresql,
both are native `sqlalchemy.ARRAY`s, while on SQLite, they are `PickleType`s.

In order to have graphene_sqlalchemy dump the arrays as arrays and not strings,
we need to subclass `PickleType`, and then use
`convert_sqlalchemy_type.register` much in the same way that graphene_sqlalchemy
does it internally for its other types.
"""
from typing import Optional, Type, Union
import sqlalchemy as sa

from ert_storage.database import IS_POSTGRES

import graphene
from graphene_sqlalchemy.converter import convert_sqlalchemy_type
from graphene_sqlalchemy.registry import Registry


__all__ = ["FloatArray", "StringArray", "IntArray"]


SQLAlchemyColumn = Union[sa.types.TypeEngine, Type[sa.types.TypeEngine]]
FloatArray: SQLAlchemyColumn
StringArray: SQLAlchemyColumn
IntArray: SQLAlchemyColumn

if IS_POSTGRES:
    FloatArray = sa.ARRAY(sa.FLOAT)
    StringArray = sa.ARRAY(sa.String)
    IntArray = sa.ARRAY(sa.Integer)
else:
    FloatArray = type("FloatArray", (sa.PickleType,), dict(sa.PickleType.__dict__))
    StringArray = type("StringArray", (sa.PickleType,), dict(sa.PickleType.__dict__))
    IntArray = type("IntArray", (sa.PickleType,), dict(sa.PickleType.__dict__))

    @convert_sqlalchemy_type.register(StringArray)
    def convert_column_to_string_array(
        type: SQLAlchemyColumn, column: sa.Column, registry: Optional[Registry] = None
    ) -> graphene.types.structures.Structure:
        return graphene.List(graphene.String)

    @convert_sqlalchemy_type.register(FloatArray)
    def convert_column_to_float_array(
        type: SQLAlchemyColumn, column: sa.Column, registry: Optional[Registry] = None
    ) -> graphene.types.structures.Structure:
        return graphene.List(graphene.Float)

    @convert_sqlalchemy_type.register(IntArray)
    def convert_column_to_int_array(
        type: SQLAlchemyColumn, column: sa.Column, registry: Optional[Registry] = None
    ) -> graphene.types.structures.Structure:
        return graphene.List(graphene.Int)
