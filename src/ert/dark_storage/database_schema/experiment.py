import sqlalchemy as sa
from uuid import uuid4
from typing import List
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ert_storage.database import Base
from ert_storage.ext.uuid import UUID
from ._userdata_field import UserdataField


class Experiment(Base, UserdataField):
    __tablename__ = "experiment"

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4, nullable=False)
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    time_updated = sa.Column(
        sa.DateTime, server_default=func.now(), onupdate=func.now()
    )
    name = sa.Column(sa.String)
    ensembles = relationship(
        "Ensemble",
        foreign_keys="[Ensemble.experiment_pk]",
        cascade="all, delete-orphan",
    )
    observations = relationship(
        "Observation",
        foreign_keys="[Observation.experiment_pk]",
        cascade="all, delete-orphan",
        back_populates="experiment",
    )
    priors = relationship(
        "Prior",
        foreign_keys="[Prior.experiment_pk]",
        cascade="all, delete-orphan",
        back_populates="experiment",
    )

    @property
    def ensemble_ids(self) -> List[UUID]:
        return [ens.id for ens in self.ensembles]
