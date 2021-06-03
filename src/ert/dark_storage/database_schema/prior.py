from enum import Enum
import sqlalchemy as sa
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from uuid import uuid4
from ert_storage.database import Base
from ert_storage.ext.uuid import UUID
from ert_storage.ext.sqlalchemy_arrays import StringArray, FloatArray
from ._userdata_field import UserdataField


class PriorFunction(Enum):
    const = 1
    trig = 2
    normal = 3
    lognormal = 4
    ert_truncnormal = 5
    stdnormal = 6
    uniform = 7
    ert_duniform = 8
    loguniform = 9
    ert_erf = 10
    ert_derf = 11


class Prior(Base, UserdataField):
    __tablename__ = "prior"

    pk = sa.Column(sa.Integer, primary_key=True)
    id = sa.Column(UUID, unique=True, default=uuid4)
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    time_updated = sa.Column(
        sa.DateTime, server_default=func.now(), onupdate=func.now()
    )

    name = sa.Column(sa.String, nullable=False)
    function = sa.Column(sa.Enum(PriorFunction), nullable=False)
    argument_names = sa.Column(StringArray, nullable=False)
    argument_values = sa.Column(FloatArray, nullable=False)

    experiment_pk = sa.Column(
        sa.Integer, sa.ForeignKey("experiment.pk"), nullable=False
    )
    experiment = relationship("Experiment")
