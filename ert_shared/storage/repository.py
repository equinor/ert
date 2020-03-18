from ert_shared.storage import (
    Observation,
    Realization,
    Ensemble,
    Base,
    ResponseDefinition,
    Response,
    ParameterDefinition,
    Parameter,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Bundle
from ert_shared.storage.session import session_factory


class ErtRepository:
    def __init__(self, session=None):

        if session is None:
            self._session = session_factory.get_session()
        else:
            self._session = session

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def commit(self):
        self._session.commit()

    def rollback(self):
        self._session.rollback()

    def close(self):
        self._session.close()

    def get_ensemble(self, name):
        return self._session.query(Ensemble).filter_by(name=name).first()

    def get_realization(self, index, ensemble_name):
        ensemble = self.get_ensemble(name=ensemble_name)
        return (
            self._session.query(Realization)
            .filter_by(ensemble_id=ensemble.id, index=index)
            .first()
        )

    def _get_response_definition(self, name, ensemble_id):
        return (
            self._session.query(ResponseDefinition)
            .filter_by(name=name, ensemble_id=ensemble_id)
            .first()
        )

    def _get_parameter_definition(self, name, group, ensemble_id):
        return (
            self._session.query(ParameterDefinition)
            .filter_by(name=name, group=group, ensemble_id=ensemble_id)
            .first()
        )

    def get_response(self, name, realization_index, ensemble_name):
        realization = self.get_realization(
            index=realization_index, ensemble_name=ensemble_name
        )
        response_definition = self._get_response_definition(
            name=name, ensemble_id=realization.ensemble.id
        )
        return (
            self._session.query(Response)
            .filter_by(
                realization_id=realization.id,
                response_definition_id=response_definition.id,
            )
            .first()
        )

    def get_response_data(self, name, ensemble_name):
        """Load lightweight "bundle" objects using the ORM."""

        bundle = Bundle("response", Response.id, Response.values, Realization.index)
        ensemble = self.get_ensemble(ensemble_name)
        response_definition = self._get_response_definition(
            name=name, ensemble_id=ensemble.id
        )
        for row in (
            self._session.query(bundle)
            .filter_by(response_definition_id=response_definition.id,)
            .join(Realization)
            .yield_per(1)
        ):
            yield row.response

    def get_parameter(self, name, group, realization_index, ensemble_name):
        realization = self.get_realization(
            index=realization_index, ensemble_name=ensemble_name
        )
        parameter_definition = self._get_parameter_definition(
            name=name, group=group, ensemble_id=realization.ensemble.id
        )
        return (
            self._session.query(Parameter)
            .filter_by(
                parameter_definition_id=parameter_definition.id,
                realization_id=realization.id,
            )
            .first()
        )

    def get_observation(self, name):
        return self._session.query(Observation).filter_by(name=name).first()

    def add_ensemble(self, name):
        ensemble = Ensemble(name=name)
        self._session.add(ensemble)
        return ensemble

    def add_realization(self, index, ensemble_name):
        ensemble = self.get_ensemble(name=ensemble_name)

        realization = Realization(index=index)
        ensemble.realizations.append(realization)

        self._session.add(realization)

        return realization

    def add_response_definition(
        self, name, indexes, ensemble_name, observation_name=None,
    ):
        ensemble = self.get_ensemble(name=ensemble_name)
        observation = None
        if observation_name is not None:
            observation = self.get_observation(name=observation_name)

        response_definition = ResponseDefinition(
            name=name,
            indexes=indexes,
            ensemble_id=ensemble.id,
            observation_id=observation.id if observation is not None else None,
        )
        self._session.add(response_definition)

        return response_definition

    def add_response(
        self, name, values, realization_index, ensemble_name,
    ):
        realization = self.get_realization(
            index=realization_index, ensemble_name=ensemble_name
        )
        response_definition = self._get_response_definition(
            name=name, ensemble_id=realization.ensemble.id
        )
        response = Response(
            values=values,
            realization_id=realization.id,
            response_definition_id=response_definition.id,
        )
        self._session.add(response)

        return response

    def add_parameter_definition(
        self, name, group, ensemble_name,
    ):
        ensemble = self.get_ensemble(name=ensemble_name)

        parameter_definition = ParameterDefinition(
            name=name, group=group, ensemble_id=ensemble.id,
        )
        self._session.add(parameter_definition)

        return parameter_definition

    def add_parameter(self, name, group, value, realization_index, ensemble_name):
        realization = self.get_realization(
            index=realization_index, ensemble_name=ensemble_name
        )

        parameter_definition = self._get_parameter_definition(
            name=name, group=group, ensemble_id=realization.ensemble.id
        )
        parameter = Parameter(
            value=value,
            realization_id=realization.id,
            parameter_definition_id=parameter_definition.id,
        )
        self._session.add(parameter)

        return parameter

    def add_observation(self, name, key_indexes, data_indexes, values, stds):
        observation = Observation(
            name=name,
            key_indexes=key_indexes,
            data_indexes=data_indexes,
            values=values,
            stds=stds,
        )
        self._session.add(observation)

        return observation

    def get_all_observation_keys(self):
        return [obs.name for obs in self._session.query(Observation.name).all()]

    def get_all_ensembles(self):
        return [ensemble for ensemble in self._session.query(Ensemble).all()]
