import sqlalchemy as sa
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.sql import func


Entity = declarative_base(name="Entity")


class Project(Entity):
    __tablename__ = "project"
    __table_args__ = (UniqueConstraint("name", name="uq_project_name"),)

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)

    ensembles = relationship("Ensemble")


class Ensemble(Entity):
    __tablename__ = "ensemble"
    __table_args__ = (
        UniqueConstraint("name", "time_created", name="uq_ensemble_name_time_created"),
    )

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)
    project_id = sa.Column(sa.Integer, sa.ForeignKey("project.id"), nullable=True)
    project = relationship("Project", back_populates="ensembles")
    time_created = sa.Column(sa.DateTime, server_default=func.now())
    num_realizations = sa.Column(sa.Integer, nullable=False)

    children = relationship(
        "Update",
        foreign_keys="[Update.ensemble_reference_id]",
    )
    parent = relationship(
        "Update",
        uselist=False,
        foreign_keys="[Update.ensemble_result_id]",
    )
    response_definitions = relationship(
        "ResponseDefinition",
        back_populates="ensemble",
    )
    parameters = relationship("Parameter", back_populates="ensemble")


class Update(Entity):
    __tablename__ = "update"
    __table_args__ = (
        UniqueConstraint("ensemble_result_id", name="uq_update_result_id"),
    )

    id = sa.Column(sa.Integer, primary_key=True)
    algorithm = sa.Column(sa.String, nullable=False)
    ensemble_reference_id = sa.Column(
        sa.Integer, sa.ForeignKey("ensemble.id"), nullable=True
    )
    ensemble_result_id = sa.Column(
        sa.Integer, sa.ForeignKey("ensemble.id"), nullable=True
    )

    ensemble_reference = relationship(
        "Ensemble",
        foreign_keys=[ensemble_reference_id],
        back_populates="children",
    )
    ensemble_result = relationship(
        "Ensemble",
        foreign_keys=[ensemble_result_id],
        uselist=False,
        back_populates="parent",
    )


class ResponseDefinition(Entity):
    __tablename__ = "response_definition"
    __table_args__ = (
        UniqueConstraint(
            "name", "ensemble_id", name="uq_response_definiton_name_ensemble_id"
        ),
    )

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)
    indices = sa.Column(sa.PickleType)
    ensemble_id = sa.Column(sa.Integer, sa.ForeignKey("ensemble.id"), nullable=False)

    ensemble = relationship("Ensemble", back_populates="response_definitions")
    observation_links = relationship(
        "ObservationResponseDefinitionLink",
        back_populates="response_definition",
    )
    responses = relationship("Response", back_populates="response_definition")


class Response(Entity):
    __tablename__ = "response"
    __table_args__ = (
        UniqueConstraint(
            "response_definition_id",
            "index",
            name="uq_response_definition_id_index",
        ),
    )

    id = sa.Column(sa.Integer, primary_key=True)
    values = sa.Column(sa.PickleType)
    index = sa.Column(sa.Integer, nullable=False)
    response_definition_id = sa.Column(
        sa.Integer, sa.ForeignKey("response_definition.id"), nullable=False
    )

    response_definition = relationship("ResponseDefinition", back_populates="responses")
    misfits = relationship("Misfit", back_populates="response")


prior_ensemble_association_table = sa.Table(
    "prior_ensemble_association_table",
    Entity.metadata,
    sa.Column("prior_id", sa.String, sa.ForeignKey("parameter_prior.id")),
    sa.Column("ensemble_id", sa.Integer, sa.ForeignKey("ensemble.id")),
)


class ParameterPrior(Entity):
    __tablename__ = "parameter_prior"

    id = sa.Column(sa.Integer, primary_key=True)
    group = sa.Column("group", sa.String)
    key = sa.Column("key", sa.String, nullable=False)
    function = sa.Column("function", sa.String)
    parameter_names = sa.Column("parameter_names", sa.PickleType)
    parameter_values = sa.Column("parameter_values", sa.PickleType)

    ensemble = relationship(
        "Ensemble", secondary=lambda: prior_ensemble_association_table, backref="priors"
    )


class Parameter(Entity):
    __tablename__ = "parameter"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "group",
            "ensemble_id",
            name="uq_parameter_name_group_ensemble_id",
        ),
    )

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)
    group = sa.Column(sa.String, nullable=False)
    ensemble_id = sa.Column(sa.Integer, sa.ForeignKey("ensemble.id"), nullable=False)
    prior_id = sa.Column(sa.Integer, sa.ForeignKey("parameter_prior.id"))

    values = sa.Column(sa.PickleType, nullable=False)

    ensemble = relationship("Ensemble", back_populates="parameters")
    prior = relationship("ParameterPrior")


class AttributeValue(Entity):
    __tablename__ = "attribute_value"

    id = sa.Column(sa.Integer, primary_key=True)
    value = sa.Column("value", sa.String, nullable=False)

    def __init__(self, value):
        self.value = value


class Observation(Entity):
    __tablename__ = "observation"
    __table_args__ = (UniqueConstraint("name", name="uq_observation_name"),)

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)
    x_axis = sa.Column(sa.PickleType)
    values = sa.Column(sa.PickleType)
    errors = sa.Column(sa.PickleType)

    attributes = association_proxy(
        "observation_attributes",
        "value",
        creator=lambda a, v: ObservationAttribute(attribute=a, value=v),
    )

    def add_attribute(self, attribute, value):
        self.attributes[attribute] = AttributeValue(value)

    def get_attribute(self, attribute):
        return self.attributes[attribute].value

    def get_attributes(self):
        return {k: v.value for k, v in self.attributes.items()}

    response_definition_links = relationship(
        "ObservationResponseDefinitionLink",
        back_populates="observation",
    )


class ObservationAttribute(Entity):
    __tablename__ = "observation_attribute"

    observation_id = sa.Column(
        sa.Integer, sa.ForeignKey("observation.id"), primary_key=True
    )
    attribute = sa.Column(sa.String)
    value_id = sa.Column(
        sa.Integer, sa.ForeignKey("attribute_value.id"), primary_key=True
    )

    value = relationship("AttributeValue")
    observation = relationship(
        Observation,
        backref=backref(
            "observation_attributes",
            collection_class=attribute_mapped_collection("attribute"),
            cascade="all, delete-orphan",
        ),
    )


class ObservationResponseDefinitionLink(Entity):
    __tablename__ = "observation_response_definition_link"
    __table_args__ = (
        UniqueConstraint(
            "response_definition_id",
            "observation_id",
            "update_id",
            name="uq_observation_response_definition_link_response_definition_id_observation_id_update_id",
        ),
    )

    id = sa.Column(sa.Integer, primary_key=True)
    response_definition_id = sa.Column(
        sa.Integer, sa.ForeignKey("response_definition.id"), nullable=False
    )
    active = sa.Column(sa.PickleType)
    observation_id = sa.Column(sa.Integer, sa.ForeignKey("observation.id"))
    update_id = sa.Column(sa.Integer, sa.ForeignKey("update.id"))

    response_definition = relationship(
        "ResponseDefinition", back_populates="observation_links"
    )
    observation = relationship(
        "Observation", back_populates="response_definition_links"
    )
    misfits = relationship(
        "Misfit", back_populates="observation_response_definition_link"
    )


class Misfit(Entity):
    __tablename__ = "misfit"
    __table_args__ = (
        UniqueConstraint(
            "response_id",
            "observation_response_definition_link_id",
            name="uq_misfit_response_id_observation_response_definition_link_id",
        ),
    )

    id = sa.Column(sa.Integer, primary_key=True)
    response_id = sa.Column(sa.Integer, sa.ForeignKey("response.id"), nullable=False)
    observation_response_definition_link_id = sa.Column(
        sa.Integer, sa.ForeignKey("observation_response_definition_link.id")
    )
    value = sa.Column(sa.Float)

    response = relationship("Response", back_populates="misfits")
    observation_response_definition_link = relationship(
        "ObservationResponseDefinitionLink", back_populates="misfits"
    )


class ObservationTransformation(Entity):
    __tablename__ = "observation_transformation"
    id = sa.Column(sa.Integer, primary_key=True)
    active_list = sa.Column(sa.PickleType, nullable=False)
    scale_list = sa.Column(sa.PickleType, nullable=False)

    observation_id = sa.Column(
        sa.Integer, sa.ForeignKey("observation.id"), nullable=False
    )
    update_id = sa.Column(sa.Integer, sa.ForeignKey("update.id"), nullable=False)

    observation = relationship("Observation")
    update = relationship("Update")
