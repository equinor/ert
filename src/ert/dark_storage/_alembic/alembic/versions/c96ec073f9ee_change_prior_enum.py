"""Change Prior Enum

Revision ID: c96ec073f9ee
Revises: 56177419dc39
Create Date: 2021-04-28 09:05:08.830307

"""
from alembic import op
import sqlalchemy as sa
from enum import Enum


# revision identifiers, used by Alembic.
revision = "c96ec073f9ee"
down_revision = "56177419dc39"
branch_labels = None
depends_on = None


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


def upgrade():
    op.drop_column("prior", "function")
    op.execute("DROP TYPE priorfunction")

    enum_type = sa.dialects.postgresql.ENUM(PriorFunction, name="priorfunction")
    enum_type.create(op.get_bind())

    op.add_column(
        "prior",
        sa.Column(
            "function",
            sa.Enum(
                PriorFunction,
            ),
            nullable=False,
        ),
    )


def downgrade():
    raise NotImplementedError("Downgrade not implemented")
