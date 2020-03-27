from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, PickleType, DateTime
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.sql import func


Entities = declarative_base(name="Entities")
Blobs = declarative_base(name="Blobs")


class Project(Entities):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String)

    __table_args__ = (UniqueConstraint("name", name="_uc_project_name_"),)

    def __repr__(self):
        return "<Project(name='{}')>".format(self.name)


class Ensemble(Entities):
    __tablename__ = "ensembles"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    project_id = Column(Integer, ForeignKey("projects.id"))
    project = relationship("Project", back_populates="ensembles")
    time_created = Column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("name", "time_created", name="_uc_name_time_"),)

    def __repr__(self):
        return "<Ensemble(name='{}')>".format(self.name)


Project.ensembles = relationship(
    "Ensemble", order_by=Ensemble.id, back_populates="project"
)


class Update(Entities):
    __tablename__ = "updates"

    id = Column(Integer, primary_key=True)
    algorithm = Column(String)
    ensemble_reference_id = Column(Integer, ForeignKey("ensembles.id"))
    ensemble_reference = relationship(
        "Ensemble", foreign_keys=[ensemble_reference_id], back_populates="children",
    )
    ensemble_result_id = Column(Integer, ForeignKey("ensembles.id"))
    ensemble_result = relationship(
        "Ensemble",
        foreign_keys=[ensemble_result_id],
        uselist=False,
        back_populates="parent",
    )

    __table_args__ = (UniqueConstraint("ensemble_result_id", name="_uc_result_id_"),)

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
    back_populates="ensemble_result",
    foreign_keys=[Update.ensemble_result_id],
)


class Realization(Entities):
    __tablename__ = "realizations"

    id = Column(Integer, primary_key=True)
    index = Column(Integer)
    ensemble_id = Column(Integer, ForeignKey("ensembles.id"))
    ensemble = relationship("Ensemble", back_populates="realizations")

    __table_args__ = (
        UniqueConstraint("index", "ensemble_id", name="_index_ensemble_id_"),
    )

    def __repr__(self):
        return "<Realization(index='{}')>".format(self.index)


Ensemble.realizations = relationship(
    "Realization", order_by=Realization.id, back_populates="ensemble"
)


class ResponseDefinition(Entities):
    __tablename__ = "response_definitions"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    indexes_ref = Column(Integer)
    ensemble_id = Column(Integer, ForeignKey("ensembles.id"))
    ensemble = relationship("Ensemble", back_populates="response_definitions")
    observation_id = Column(Integer, ForeignKey("observations.id"))
    observation = relationship("Observation", back_populates="response_definitions")

    __table_args__ = (
        UniqueConstraint("name", "ensemble_id", name="_name_ensemble_id_"),
    )

    def __repr__(self):
        return "<ResponseDefinition(name='{}', indexes_ref='{}', ensemble_id='{}')>".format(
            self.name, self.indexes_ref, self.ensemble_id
        )


Ensemble.response_definitions = relationship(
    "ResponseDefinition", order_by=ResponseDefinition.id, back_populates="ensemble",
)


class Response(Entities):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True)
    values_ref = Column(Integer)
    realization_id = Column(Integer, ForeignKey("realizations.id"))
    realization = relationship("Realization", back_populates="responses")
    response_definition_id = Column(Integer, ForeignKey("response_definitions.id"))
    response_definition = relationship("ResponseDefinition", back_populates="responses")

    __table_args__ = (
        UniqueConstraint(
            "realization_id", "response_definition_id", name="_uc_realization_reponse_"
        ),
    )

    def __repr__(self):
        return "<Response(values_ref='{}', realization_id='{}', response_definition_id='{}')>".format(
            self.values_ref,
            self.realization_id,
            self.response_definition_id,
        )


Realization.responses = relationship(
    "Response", order_by=Response.id, back_populates="realization"
)
ResponseDefinition.responses = relationship(
    "Response", order_by=Response.id, back_populates="response_definition"
)


class ParameterDefinition(Entities):
    __tablename__ = "parameter_definitions"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    group = Column(String)
    ensemble_id = Column(Integer, ForeignKey("ensembles.id"))
    ensemble = relationship("Ensemble", back_populates="parameter_definitions")

    __table_args__ = (
        UniqueConstraint("name", "group", "ensemble_id", name="_name_ensemble_id_"),
    )

    def __repr__(self):
        return "<ParameterDefinition(name='{}', group='{}', ensemble_id='{}')>".format(
            self.name, self.group, self.ensemble_id
        )


Ensemble.parameter_definitions = relationship(
    "ParameterDefinition", order_by=ParameterDefinition.id, back_populates="ensemble"
)


class Parameter(Entities):
    __tablename__ = "parameters"

    id = Column(Integer, primary_key=True)
    value_ref = Column(Integer)
    realization_id = Column(Integer, ForeignKey("realizations.id"))
    realization = relationship("Realization", back_populates="parameters")
    parameter_definition_id = Column(Integer, ForeignKey("parameter_definitions.id"))
    parameter_definition = relationship(
        "ParameterDefinition", back_populates="parameters"
    )

    __table_args__ = (
        UniqueConstraint(
            "realization_id",
            "parameter_definition_id",
            name="_uc_realization_parameter_",
        ),
    )

    def __repr__(self):
        return "<Parameter(value_ref='{}', realization_id='{}', parameter_definition_id='{}')>".format(
            self.value_ref, self.realization_id, self.parameter_definition_id
        )


Realization.parameters = relationship(
    "Parameter", order_by=Parameter.id, back_populates="realization"
)
ParameterDefinition.parameters = relationship(
    "Parameter", order_by=Parameter.id, back_populates="parameter_definition"
)


class Observation(Entities):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    key_indexes_ref = Column(Integer)
    data_indexes_ref = Column(Integer)
    values_ref = Column(Integer)
    stds_ref = Column(Integer)

    __table_args__ = (UniqueConstraint("name", name="_uc_observation_name_"),)

    def __repr__(self):
        return "<Observation(name='{}', key_indexes_ref='{}', data_indexes_ref='{}', values_ref='{}', stds_ref='{}')>".format(
            self.name,
            self.key_indexes_ref,
            self.data_indexes_ref,
            self.values_ref,
            self.stds_ref,
        )


Observation.response_definitions = relationship(
    "ResponseDefinition", order_by=ResponseDefinition.id, back_populates="observation"
)


class ErtBlob(Blobs):
    __tablename__ = "ert_blobs"

    id = Column(Integer, primary_key=True)
    data = Column(PickleType)

    def __repr__(self):
        return "<Value(id='{}', data='{}')>".format(self.id, self.data)
