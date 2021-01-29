"""Remove Realizations

Revision ID: 22487ba09fe3
Revises: dbe9b8d6288b
Create Date: 2021-01-28 09:48:22.892822

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.declarative import declarative_base


# revision identifiers, used by Alembic.
revision = "22487ba09fe3"
down_revision = "dbe9b8d6288b"
branch_labels = None
depends_on = None


Base = declarative_base()


class Ensemble(Base):
    __tablename__ = "ensemble"

    id = sa.Column(sa.Integer, primary_key=True)
    num_realizations = sa.Column(sa.Integer, nullable=False)


class Realization(Base):
    __tablename__ = "realization"

    id = sa.Column(sa.Integer, primary_key=True)
    index = sa.Column(sa.Integer, nullable=False)
    ensemble_id = sa.Column(sa.Integer, sa.ForeignKey("ensemble.id"), nullable=False)


class Response(Base):
    __tablename__ = "response"

    id = sa.Column(sa.Integer, primary_key=True)
    index = sa.Column(sa.Integer, nullable=False)
    realization_id = sa.Column(
        sa.Integer, sa.ForeignKey("realization.id"), nullable=False
    )


def upgrade():
    session = orm.Session(bind=op.get_bind())

    # 1. Add column to ensemble, response
    op.add_column(
        "ensemble", sa.Column("num_realizations", sa.Integer(), nullable=True)
    )
    op.add_column("response", sa.Column("index", sa.Integer(), nullable=True))

    # 2. Count ensembles' realizations, copy index from realizations to responses
    for ens in session.query(Ensemble):
        ens.num_realizations = (
            session.query(Realization).filter_by(ensemble_id=ens.id).count()
        )
    session.commit()

    for resp in session.query(Response):
        resp.index = (
            session.query(Realization.index)
            .filter_by(id=resp.realization_id)
            .one()
            .index
        )
    session.commit()

    # 3. Finalise
    with op.batch_alter_table("response") as bop:
        bop.drop_constraint(
            "uq_response_realization_id_reponse_defition_id", type_="unique"
        )
        bop.create_unique_constraint(
            "uq_response_definition_id_index",
            ["response_definition_id", "index"],
        )
        bop.drop_column("realization_id")
    op.drop_table("realization")

    # 4. Set nullable
    with op.batch_alter_table("ensemble") as bop:
        bop.alter_column("num_realizations", existing_type=sa.Integer(), nullable=False)
    with op.batch_alter_table("response") as bop:
        bop.alter_column("index", existing_type=sa.Integer(), nullable=False)


def downgrade():
    sys.exit("Cannot downgrade")
