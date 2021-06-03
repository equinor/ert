import sqlalchemy as sa
from ert_storage.database import Base
from ert_storage.ext.sqlalchemy_arrays import StringArray, FloatArray
from ert_storage.ext.uuid import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from uuid import uuid4
from ._userdata_field import UserdataField

observation_record_association = sa.Table(
    "observation_record_association",
    Base.metadata,
    sa.Column("observation_pk", sa.Integer, sa.ForeignKey("observation.pk")),
    sa.Column("record_pk", sa.Integer, sa.ForeignKey("record.pk")),
)


class Observation(Base, UserdataField):
    __tablename__ = "observation"
    __table_args__ = (
        sa.UniqueConstraint("name", "experiment_pk", name="uq_observation_name"),
    )

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4, nullable=False)
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    time_updated = sa.Column(
        sa.DateTime, server_default=func.now(), onupdate=func.now()
    )
    name = sa.Column(sa.String, nullable=False)
    x_axis = sa.Column(StringArray, nullable=False)
    values = sa.Column(FloatArray, nullable=False)
    errors = sa.Column(FloatArray, nullable=False)

    records = relationship(
        "Record",
        secondary=observation_record_association,
        back_populates="observations",
    )
    experiment_pk = sa.Column(
        sa.Integer, sa.ForeignKey("experiment.pk"), nullable=False
    )
    experiment = relationship("Experiment")


class ObservationTransformation(Base):
    __tablename__ = "observation_transformation"

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4, nullable=False)
    active_list = sa.Column(sa.PickleType, nullable=False)
    scale_list = sa.Column(sa.PickleType, nullable=False)

    observation_pk = sa.Column(
        sa.Integer, sa.ForeignKey("observation.pk"), nullable=False
    )
    observation = relationship("Observation")

    update_pk = sa.Column(sa.Integer, sa.ForeignKey("update.pk"), nullable=False)
    update = relationship("Update", back_populates="observation_transformations")
