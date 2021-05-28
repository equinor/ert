from ert_storage.database import Base
import sqlalchemy as sa
from sqlalchemy.orm import relationship
from uuid import uuid4
from ert_storage.ext.uuid import UUID


class Update(Base):
    __tablename__ = "update"
    __table_args__ = (
        sa.UniqueConstraint("ensemble_result_pk", name="uq_update_result_pk"),
    )

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4, nullable=False)
    algorithm = sa.Column(sa.String, nullable=False)
    ensemble_reference_pk = sa.Column(
        sa.Integer, sa.ForeignKey("ensemble.pk"), nullable=True
    )
    ensemble_result_pk = sa.Column(
        sa.Integer, sa.ForeignKey("ensemble.pk"), nullable=True
    )

    ensemble_reference = relationship(
        "Ensemble",
        foreign_keys=[ensemble_reference_pk],
        back_populates="children",
    )
    ensemble_result = relationship(
        "Ensemble",
        foreign_keys=[ensemble_result_pk],
        uselist=False,
        back_populates="parent",
    )
    observation_transformations = relationship(
        "ObservationTransformation",
        foreign_keys="[ObservationTransformation.update_pk]",
        cascade="all, delete-orphan",
    )
