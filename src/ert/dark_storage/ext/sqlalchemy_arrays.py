"""
This module adds the FloatArray, StringArray and IntArray column types. In Postgresql,
both are native `sqlalchemy.ARRAY`s, while on SQLite, they are `PickleType`s.

In order to have graphene_sqlalchemy dump the arrays as arrays and not strings,
we need to subclass `PickleType`, and then use
`convert_sqlalchemy_type.register` much in the same way that graphene_sqlalchemy
does it internally for its other types.
"""
from typing import Type, Union
import sqlalchemy as sa

from ert_storage.database import IS_POSTGRES

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
