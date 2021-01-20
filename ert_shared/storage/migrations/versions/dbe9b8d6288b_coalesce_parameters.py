"""Coalesce Parameters

Revision ID: dbe9b8d6288b
Revises: 22d6bbf0a926
Create Date: 2021-01-19 21:41:34.906112

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.declarative import declarative_base


# revision identifiers, used by Alembic.
revision = "dbe9b8d6288b"
down_revision = "22d6bbf0a926"
branch_labels = None
depends_on = None


Base = declarative_base()


class Ensemble(Base):
    __tablename__ = "ensemble"

    id = sa.Column(sa.Integer, primary_key=True)


class Realization(Base):
    __tablename__ = "realization"

    id = sa.Column(sa.Integer, primary_key=True)
    index = sa.Column(sa.Integer, nullable=False)
    ensemble_id = sa.Column(sa.Integer, sa.ForeignKey("ensemble.id"), nullable=False)

    parameters = orm.relationship("Parameter", back_populates="realization")


class ParameterDefinition(Base):
    __tablename__ = "parameter_definition"

    id = sa.Column(sa.Integer, primary_key=True)
    values = sa.Column(sa.PickleType)
    ensemble_id = sa.Column(sa.Integer, sa.ForeignKey("ensemble.id"), nullable=False)

    parameters = orm.relationship("Parameter")


class Parameter(Base):
    __tablename__ = "parameter"

    id = sa.Column(sa.Integer, primary_key=True)
    value = sa.Column(sa.PickleType)
    realization_id = sa.Column(
        sa.Integer, sa.ForeignKey("realization.id"), nullable=False
    )
    parameter_definition_id = sa.Column(
        sa.Integer, sa.ForeignKey("parameter_definition.id"), nullable=False
    )
    realization = orm.relationship("Realization", back_populates="parameters")


def upgrade():
    session = orm.Session(bind=op.get_bind())

    # 1. Add column to parameter_definition
    op.add_column(
        "parameter_definition", sa.Column("values", sa.PickleType(), nullable=True)
    )

    # 2. Copy data from parameters
    for paramdef in session.query(ParameterDefinition):
        realization_count = (
            session.query(Realization)
            .filter_by(ensemble_id=paramdef.ensemble_id)
            .count()
        )

        values = [None] * realization_count
        for param in paramdef.parameters:
            values[param.realization.index] = param.value
        paramdef.values = values
    session.commit()

    # 3. Finalise
    op.drop_table("parameter")
    op.rename_table("parameter_definition", "parameter")
    with op.batch_alter_table("parameter") as bop:
        bop.alter_column("values", nullable=True)


def downgrade():
    sys.exit("Cannot downgrade")
