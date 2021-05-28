from enum import Enum

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ert_storage.database import Base


class RecordType(Enum):
    f64_matrix = 1
    file = 2


class RecordClass(Enum):
    parameter = 1
    response = 2
    other = 3


class RecordInfo(Base):
    __tablename__ = "record_info"
    __table_args__ = (sa.UniqueConstraint("name", "ensemble_pk"),)

    pk = sa.Column(sa.Integer, primary_key=True)
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    time_updated = sa.Column(
        sa.DateTime, server_default=func.now(), onupdate=func.now()
    )

    ensemble_pk = sa.Column(sa.Integer, sa.ForeignKey("ensemble.pk"), nullable=False)
    ensemble = relationship("Ensemble")

    records = relationship(
        "Record",
        foreign_keys="[Record.record_info_pk]",
        cascade="all, delete-orphan",
    )

    name = sa.Column(sa.String, nullable=False)
    record_type = sa.Column(sa.Enum(RecordType), nullable=False)
    record_class = sa.Column(sa.Enum(RecordClass), nullable=False)

    # Parameter-specific data
    prior_pk = sa.Column(sa.Integer, sa.ForeignKey("prior.pk"), nullable=True)
    prior = relationship("Prior")
