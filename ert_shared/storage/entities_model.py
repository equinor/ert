from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PickleType,
    String,
    Table,
)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.schema import UniqueConstraint, MetaData
from sqlalchemy.sql import func

meta = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

Entities = declarative_base(name="Entities", metadata=meta)


class Project(Entities):
    __tablename__ = "project"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("name", name="uq_project_name"),)

    def __repr__(self):
        return "<Project(name='{}')>".format(self.name)


class Ensemble(Entities):
    __tablename__ = "ensemble"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    project_id = Column(Integer, ForeignKey("project.id"), nullable=True)
    project = relationship("Project", back_populates="ensembles")
    time_created = Column(DateTime, server_default=func.now())
    __table_args__ = (
        UniqueConstraint("name", "time_created", name="uq_ensemble_name_time_created"),
    )

    def __repr__(self):
        return "<Ensemble(name='{}')>".format(self.name)


Project.ensembles = relationship(
    "Ensemble", order_by=Ensemble.id, back_populates="project"
)


class Update(Entities):
    __tablename__ = "update"

    id = Column(Integer, primary_key=True)
    algorithm = Column(String, nullable=False)
    ensemble_reference_id = Column(Integer, ForeignKey("ensemble.id"), nullable=False)
    ensemble_reference = relationship(
        "Ensemble",
        foreign_keys=[ensemble_reference_id],
        back_populates="children",
    )
    ensemble_result_id = Column(Integer, ForeignKey("ensemble.id"), nullable=False)
    ensemble_result = relationship(
        "Ensemble",
        foreign_keys=[ensemble_result_id],
        uselist=False,
        back_populates="parent",
    )

    __table_args__ = (
        UniqueConstraint("ensemble_result_id", name="uq_update_result_id"),
    )

    def __repr__(self):
        return "<Update(algorithm='{}', ensemble_reference_id='{}', ensemble_result_id='{}')>".format(
            self.algorithm, self.ensemble_reference_id, self.ensemble_result_id
        )


Ensemble.children = relationship(
    "Update",
    order_by=Update.id,
    back_populates="ensemble_reference",
    foreign_keys=[Update.ensemble_reference_id],
)

Ensemble.parent = relationship(
    "Update",
    order_by=Update.id,
    uselist=False,
    back_populates="ensemble_result",
    foreign_keys=[Update.ensemble_result_id],
)


class Realization(Entities):
    __tablename__ = "realization"

    id = Column(Integer, primary_key=True)
    index = Column(Integer, nullable=False)
    ensemble_id = Column(Integer, ForeignKey("ensemble.id"), nullable=False)
    ensemble = relationship("Ensemble", back_populates="realizations")

    __table_args__ = (
        UniqueConstraint(
            "index", "ensemble_id", name="uq_realization_index_ensemble_id"
        ),
    )

    def __repr__(self):
        return "<Realization(index='{}')>".format(self.index)


Ensemble.realizations = relationship(
    "Realization", order_by=Realization.id, back_populates="ensemble"
)


class ResponseDefinition(Entities):
    __tablename__ = "response_definition"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    indexes_ref = Column(Integer)  # Reference to the description of  plot axis
    indexes = Column(PickleType)
    ensemble_id = Column(Integer, ForeignKey("ensemble.id"), nullable=False)
    ensemble = relationship("Ensemble", back_populates="response_definitions")

    __table_args__ = (
        UniqueConstraint(
            "name", "ensemble_id", name="uq_response_definiton_name_ensemble_id"
        ),
    )

    def __repr__(self):
        return "<ResponseDefinition(name='{}', ensemble_id='{}')>".format(
            self.name, self.ensemble_id
        )


Ensemble.response_definitions = relationship(
    "ResponseDefinition",
    order_by=ResponseDefinition.id,
    back_populates="ensemble",
)


class Response(Entities):
    __tablename__ = "response"

    id = Column(Integer, primary_key=True)
    values_ref = Column(Integer)
    values = Column(PickleType)
    realization_id = Column(Integer, ForeignKey("realization.id"), nullable=False)
    realization = relationship("Realization", back_populates="responses")
    response_definition_id = Column(
        Integer, ForeignKey("response_definition.id"), nullable=False
    )
    response_definition = relationship("ResponseDefinition", back_populates="responses")

    __table_args__ = (
        UniqueConstraint(
            "realization_id",
            "response_definition_id",
            name="uq_response_realization_id_reponse_defition_id",
        ),
    )

    def __repr__(self):
        return "<Response(realization_id='{}', response_definition_id='{}')>".format(
            self.realization_id,
            self.response_definition_id,
        )


Realization.responses = relationship(
    "Response", order_by=Response.id, back_populates="realization"
)
ResponseDefinition.responses = relationship(
    "Response", order_by=Response.id, back_populates="response_definition"
)

prior_ensemble_association_table = Table(
    "prior_ensemble_association_table",
    Entities.metadata,
    Column("prior_id", String, ForeignKey("parameter_prior.id")),
    Column("ensemble_id", Integer, ForeignKey("ensemble.id")),
)


class ParameterPrior(Entities):
    __tablename__ = "parameter_prior"

    id = Column(Integer, primary_key=True)
    group = Column("group", String)
    key = Column("key", String, nullable=False)
    function = Column("function", String)
    parameter_names = Column("parameter_names", PickleType)
    parameter_values = Column("parameter_values", PickleType)

    ensemble = relationship(
        "Ensemble", secondary=lambda: prior_ensemble_association_table, backref="priors"
    )


class ParameterDefinition(Entities):
    __tablename__ = "parameter_definition"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    group = Column(String, nullable=False)
    ensemble_id = Column(Integer, ForeignKey("ensemble.id"), nullable=False)
    ensemble = relationship("Ensemble", back_populates="parameter_definitions")
    prior_id = Column(Integer, ForeignKey("parameter_prior.id"))
    prior = relationship("ParameterPrior")

    __table_args__ = (
        UniqueConstraint(
            "name",
            "group",
            "ensemble_id",
            name="uq_parameter_definition_name_group_ensemble_id",
        ),
    )

    def __repr__(self):
        return "<ParameterDefinition(name='{}', group='{}', ensemble_id='{}')>".format(
            self.name, self.group, self.ensemble_id
        )


Ensemble.parameter_definitions = relationship(
    "ParameterDefinition", order_by=ParameterDefinition.id, back_populates="ensemble"
)


class Parameter(Entities):
    __tablename__ = "parameter"

    id = Column(Integer, primary_key=True)
    value_ref = Column(Integer)
    value = Column(PickleType)
    realization_id = Column(Integer, ForeignKey("realization.id"), nullable=False)
    realization = relationship("Realization", back_populates="parameters")
    parameter_definition_id = Column(
        Integer, ForeignKey("parameter_definition.id"), nullable=False
    )
    parameter_definition = relationship(
        "ParameterDefinition", back_populates="parameters"
    )

    __table_args__ = (
        UniqueConstraint(
            "realization_id",
            "parameter_definition_id",
            name="uq_parameter_realization_id_parameter_definition_id",
        ),
    )

    def __repr__(self):
        return "<Parameter(realization_id='{}', parameter_definition_id='{}')>".format(
            self.realization_id, self.parameter_definition_id
        )


Realization.parameters = relationship(
    "Parameter", order_by=Parameter.id, back_populates="realization"
)
ParameterDefinition.parameters = relationship(
    "Parameter", order_by=Parameter.id, back_populates="parameter_definition"
)


class AttributeValue(Entities):
    __tablename__ = "attribute_value"

    id = Column(Integer, primary_key=True)
    value = Column("value", String, nullable=False)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "AttributeValue(%s)" % repr(self.value)


class Observation(Entities):
    __tablename__ = "observation"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    key_indexes_ref = Column(Integer)
    key_indexes = Column(PickleType)
    data_indexes_ref = Column(Integer)
    data_indexes = Column(PickleType)
    values_ref = Column(Integer)
    values = Column(PickleType)
    stds_ref = Column(Integer)
    stds = Column(PickleType)

    attributes = association_proxy(
        "observation_attributes",
        "value",
        creator=lambda a, v: ObservationAttribute(attribute=a, value=v),
    )

    __table_args__ = (UniqueConstraint("name", name="uq_observation_name"),)

    def __repr__(self):
        return "<Observation(name='{}')>".format(self.name)

    def add_attribute(self, attribute, value):
        self.attributes[attribute] = AttributeValue(value)

    def get_attribute(self, attribute):
        return self.attributes[attribute].value

    def get_attributes(self):
        return {k: v.value for k, v in self.attributes.items()}


class ObservationAttribute(Entities):
    __tablename__ = "observation_attribute"

    observation_id = Column(Integer, ForeignKey("observation.id"), primary_key=True)
    attribute = Column(String)
    value_id = Column(Integer, ForeignKey("attribute_value.id"), primary_key=True)
    value = relationship("AttributeValue")

    observation = relationship(
        Observation,
        backref=backref(
            "observation_attributes",
            collection_class=attribute_mapped_collection("attribute"),
            cascade="all, delete-orphan",
        ),
    )


class ObservationResponseDefinitionLink(Entities):
    __tablename__ = "observation_response_definition_link"

    id = Column(Integer, primary_key=True)
    response_definition_id = Column(
        Integer, ForeignKey("response_definition.id"), nullable=False
    )
    active_ref = Column(Integer)
    active = Column(PickleType)
    response_definition = relationship(
        "ResponseDefinition", back_populates="observation_links"
    )
    observation_id = Column(Integer, ForeignKey("observation.id"))
    observation = relationship(
        "Observation", back_populates="response_definition_links"
    )
    update_id = Column(Integer, ForeignKey("update.id"))
    __table_args__ = (
        UniqueConstraint(
            "response_definition_id",
            "observation_id",
            "update_id",
            name="uq_observation_response_definition_link_response_definition_id_observation_id_update_id",
        ),
    )

    def __repr__(self):
        return "<ObservationResponseDefinitionLink(response_definition_id='{}', observation_id='{}', update_id='{}')>".format(
            self.response_definition_id, self.observation_id, self.update_id
        )


ResponseDefinition.observation_links = relationship(
    "ObservationResponseDefinitionLink",
    order_by=ObservationResponseDefinitionLink.id,
    back_populates="response_definition",
)
Observation.response_definition_links = relationship(
    "ObservationResponseDefinitionLink",
    order_by=ObservationResponseDefinitionLink.id,
    back_populates="observation",
)


class Misfit(Entities):
    __tablename__ = "misfit"

    id = Column(Integer, primary_key=True)
    response_id = Column(Integer, ForeignKey("response.id"), nullable=False)
    response = relationship("Response", back_populates="misfits")
    observation_response_definition_link_id = Column(
        Integer, ForeignKey("observation_response_definition_link.id")
    )
    observation_response_definition_link = relationship(
        "ObservationResponseDefinitionLink", back_populates="misfits"
    )
    value = Column(Float)

    __table_args__ = (
        UniqueConstraint(
            "response_id",
            "observation_response_definition_link_id",
            name="uq_misfit_response_id_observation_response_definition_link_id",
        ),
    )

    def __repr__(self):
        return "<Misfit(response_id='{}', observation_response_definition_link_id='{}', misfit='{}')>".format(
            self.response_id,
            self.observation_response_definition_link_id,
            self.value,
        )


Response.misfits = relationship("Misfit", order_by=Misfit.id, back_populates="response")
ObservationResponseDefinitionLink.misfits = relationship(
    "Misfit", order_by=Misfit.id, back_populates="observation_response_definition_link"
)
