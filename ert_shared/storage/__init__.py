from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, PickleType
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.schema import UniqueConstraint


Base = declarative_base()


class Ensemble(Base):
    __tablename__ = "ensembles"

    id = Column(Integer, primary_key=True)
    name = Column(String)

    __table_args__ = (UniqueConstraint("name", name="_name_ensemble_id_"),)

    def __repr__(self):
        return "<Ensemble(name='{}')>".format(self.name)


class Realization(Base):
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


class ResponseDefinition(Base):
    __tablename__ = "response_definitions"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    indexes = Column(PickleType)
    ensemble_id = Column(Integer, ForeignKey("ensembles.id"))
    ensemble = relationship("Ensemble", back_populates="response_definitions")
    observation_id = Column(Integer, ForeignKey("observations.id"))
    observation = relationship("Observation", back_populates="response_definitions")

    __table_args__ = (
        UniqueConstraint("name", "ensemble_id", name="_name_ensemble_id_"),
    )

    def __repr__(self):
        return "<ResponseDefinition(name='{}', indexes='{}', ensemble_id='{}')>".format(
            self.name, self.indexes, self.ensemble_id
        )


Ensemble.response_definitions = relationship(
    "ResponseDefinition", order_by=ResponseDefinition.id, back_populates="ensemble",
)


class Response(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True)
    values = Column(PickleType)
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
        return "<Response(name='{}', values='{:.3f}', realization_id='{}', response_definition_id='{}')>".format(
            self.name, self.values, self.realization_id, self.response_definition_id,
        )


Realization.responses = relationship(
    "Response", order_by=Response.id, back_populates="realization"
)
ResponseDefinition.responses = relationship(
    "Response", order_by=Response.id, back_populates="response_definition"
)


class ParameterDefinition(Base):
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


class Parameter(Base):
    __tablename__ = "parameters"

    id = Column(Integer, primary_key=True)
    value = Column(Float)
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
        return "<Parameter(value='{:.3f}', realization_id='{}', parameter_definition_id='{}')>".format(
            self.value, self.realization_id, self.parameter_definition_id
        )


Realization.parameters = relationship(
    "Parameter", order_by=Parameter.id, back_populates="realization"
)
ParameterDefinition.parameters = relationship(
    "Parameter", order_by=Parameter.id, back_populates="parameter_definition"
)


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    key_indexes = Column(PickleType)
    data_indexes = Column(PickleType)
    values = Column(PickleType)
    stds = Column(PickleType)

    __table_args__ = (UniqueConstraint("name", name="_uc_observation_name_"),)

    def __repr__(self):
        return "<Observation(name='{}', key_index='{}', data_index='{}', value='{}', std='{}')>".format(
            self.name, self.key_indexes, self.data_indexes, self.values, self.stds
        )


Observation.response_definitions = relationship(
    "ResponseDefinition", order_by=ResponseDefinition.id, back_populates="observation"
)
