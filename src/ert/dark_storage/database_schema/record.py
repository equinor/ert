from typing import Any
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from ert_storage.ext.sqlalchemy_arrays import FloatArray
from ert_storage.ext.uuid import UUID
from ert_storage.database import Base

from ._userdata_field import UserdataField
from .observation import observation_record_association
from .record_info import RecordType, RecordClass


class Record(Base, UserdataField):
    __tablename__ = "record"

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4, nullable=False)
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    time_updated = sa.Column(
        sa.DateTime, server_default=func.now(), onupdate=func.now()
    )

    realization_index = sa.Column(sa.Integer, nullable=True)

    record_info_pk = sa.Column(
        sa.Integer, sa.ForeignKey("record_info.pk"), nullable=True
    )
    record_info = relationship("RecordInfo", back_populates="records")

    file_pk = sa.Column(sa.Integer, sa.ForeignKey("file.pk"))
    f64_matrix_pk = sa.Column(sa.Integer, sa.ForeignKey("f64_matrix.pk"))

    file = relationship("File", cascade="all")
    f64_matrix = relationship("F64Matrix", cascade="all")

    observations = relationship(
        "Observation",
        secondary=observation_record_association,
        back_populates="records",
    )

    @property
    def data(self) -> Any:
        info = self.record_info
        if info.record_type == RecordType.file:
            return self.file.content
        elif info.record_type == RecordType.f64_matrix:
            return self.f64_matrix.content
        else:
            raise NotImplementedError(
                f"The record type {self.record_type} is not yet implemented"
            )

    @property
    def ensemble_pk(self) -> int:
        return self.record_info.ensemble_pk

    @property
    def name(self) -> str:
        return self.record_info.name

    @property
    def record_type(self) -> RecordType:
        return self.record_info.record_type

    @property
    def record_class(self) -> RecordClass:
        return self.record_info.record_class


class File(Base):
    __tablename__ = "file"

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4, nullable=False)
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    time_updated = sa.Column(
        sa.DateTime, server_default=func.now(), onupdate=func.now()
    )

    filename = sa.Column(sa.String, nullable=False)
    mimetype = sa.Column(sa.String, nullable=False)

    content = sa.Column(sa.LargeBinary)
    az_container = sa.Column(sa.String)
    az_blob = sa.Column(sa.String)


class F64Matrix(Base):
    __tablename__ = "f64_matrix"

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4, nullable=False)
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    time_updated = sa.Column(
        sa.DateTime, server_default=func.now(), onupdate=func.now()
    )
    content = sa.Column(FloatArray, nullable=False)
    labels = sa.Column(sa.PickleType)


class FileBlock(Base):
    __tablename__ = "file_block"

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4, nullable=False)
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    time_updated = sa.Column(
        sa.DateTime, server_default=func.now(), onupdate=func.now()
    )
    block_id = sa.Column(sa.String, nullable=False)
    block_index = sa.Column(sa.Integer, nullable=False)
    record_name = sa.Column(sa.String, nullable=False)
    realization_index = sa.Column(sa.Integer, nullable=True)
    ensemble_pk = sa.Column(sa.Integer, sa.ForeignKey("ensemble.pk"), nullable=True)
    ensemble = relationship("Ensemble")
    content = sa.Column(sa.LargeBinary, nullable=True)
