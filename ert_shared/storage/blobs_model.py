from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PickleType,
    String,
    Table,
)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.schema import UniqueConstraint, MetaData
from sqlalchemy.sql import func

meta = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

Blobs = declarative_base(name="Blobs", metadata=meta)


class ErtBlob(Blobs):
    __tablename__ = "ert_blob"

    id = Column(Integer, primary_key=True)
    data = Column(PickleType)

    def __repr__(self):
        return "<Value(id='{}', data='{}')>".format(self.id, self.data)
