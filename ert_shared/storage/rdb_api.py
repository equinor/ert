import logging

from ert_shared.storage.model import (
    Ensemble,
    Observation,
    Parameter,
    ParameterDefinition,
    Realization,
    Response,
    ResponseDefinition,
    Update,
)
from ert_shared.storage.session import session_factory
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Bundle


class RdbApi:
    def __init__(self, session=None):

        if session is None:
            self._session = session_factory.get_entities_session()
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
        return self._session.query(Ensemble).filter_by(name=name).order_by(desc(Ensemble.time_created)).first()

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

    def add_ensemble(self, name, reference=None):
        msg = "Adding ensemble with name '{}'"
        logging.info(msg.format(name))

        ensemble = Ensemble(name=name)
        self._session.add(ensemble)
        if reference is not None:
            msg = "Adding ensemble '{}' as reference. '{}' is used on this update step."
            logging.info(msg.format(reference[0], reference[1]))

            reference_ensemble = self.get_ensemble(reference[0])
            update = Update(algorithm=reference[1])
            update.ensemble_reference = reference_ensemble
            update.ensemble_result = ensemble
            self._session.add(update)
        return ensemble

    def add_realization(self, index, ensemble_name):
        msg = "Adding realization with index '{}' on ensemble '{}'"
        logging.info(msg.format(index, ensemble_name))

        ensemble = self.get_ensemble(name=ensemble_name)

        realization = Realization(index=index)
        ensemble.realizations.append(realization)

        self._session.add(realization)

        return realization

    def add_response_definition(
        self, name, indexes_ref, ensemble_name, observation_name=None,
    ):
        msg = "Adding response definition with name '{}' on ensemble '{}'. Attaching indexes with ref '{}'"
        logging.info(msg.format(name, ensemble_name, indexes_ref))

        ensemble = self.get_ensemble(name=ensemble_name)
        observation = None
        if observation_name is not None:
            msg = "Connecting observation '{}'"
            logging.info(msg.format(observation_name))

            observation = self.get_observation(name=observation_name)

        response_definition = ResponseDefinition(
            name=name,
            indexes_ref=indexes_ref,
            ensemble_id=ensemble.id,
            observation_id=observation.id if observation is not None else None,
        )
        self._session.add(response_definition)

        return response_definition

    def add_response(
        self, name, values_ref, realization_index, ensemble_name,
    ):
        msg = "Adding response with name '{}' on ensemble '{}', realization '{}'. Attaching values with ref '{}'"
        logging.info(msg.format(name, ensemble_name, realization_index, values_ref))

        realization = self.get_realization(
            index=realization_index, ensemble_name=ensemble_name
        )
        response_definition = self._get_response_definition(
            name=name, ensemble_id=realization.ensemble.id
        )
        response = Response(
            values_ref=values_ref,
            realization_id=realization.id,
            response_definition_id=response_definition.id,
        )
        self._session.add(response)

        return response

    def add_parameter_definition(
        self, name, group, ensemble_name,
    ):
        msg = (
            "Adding parameter definition with name '{}' in group '{}' on ensemble '{}'"
        )
        logging.info(msg.format(name, group, ensemble_name))

        ensemble = self.get_ensemble(name=ensemble_name)

        parameter_definition = ParameterDefinition(
            name=name, group=group, ensemble_id=ensemble.id,
        )
        self._session.add(parameter_definition)

        return parameter_definition

    def add_parameter(self, name, group, value_ref, realization_index, ensemble_name):
        msg = "Adding parameter with name '{}', group '{}', realization '{}', value_ref '{}', ensemble '{}'"
        logging.info(
            msg.format(name, group, realization_index, value_ref, ensemble_name)
        )

        realization = self.get_realization(
            index=realization_index, ensemble_name=ensemble_name
        )

        parameter_definition = self._get_parameter_definition(
            name=name, group=group, ensemble_id=realization.ensemble.id
        )
        parameter = Parameter(
            value_ref=value_ref,
            realization_id=realization.id,
            parameter_definition_id=parameter_definition.id,
        )
        self._session.add(parameter)

        return parameter

    def add_observation(
        self, name, key_indexes_ref, data_indexes_ref, values_ref, stds_ref
    ):
        msg = "Adding observation with name '{}', key_indexes_ref '{}', data_indexes_ref '{}', values_ref '{}', stds_ref '{}'"
        logging.info(
            msg.format(name, key_indexes_ref, data_indexes_ref, values_ref, stds_ref)
        )

        observation = Observation(
            name=name,
            key_indexes_ref=key_indexes_ref,
            data_indexes_ref=data_indexes_ref,
            values_ref=values_ref,
            stds_ref=stds_ref,
        )
        self._session.add(observation)

        return observation

    def get_all_observation_keys(self):
        return [obs.name for obs in self._session.query(Observation.name).all()]

    def get_all_ensembles(self):
        return [ensemble for ensemble in self._session.query(Ensemble).all()]

    ############## - musiv - new functions ########################
    def get_realizations_by_ensemble_id(self, ensemble_id):
        return (
            self._session.query(Realization)
            .filter_by(ensemble_id=ensemble_id)
        )   

    def get_ensemble_by_id(self, ensemble_id):
        return (
            self._session.query(Ensemble)
            .filter_by(id=ensemble_id)
            .one()
        )

    def get_response_definitions_by_ensemble_id(self, ensemble_id):
        return (
            self._session.query(ResponseDefinition)
            .filter_by(ensemble_id=ensemble_id)
        )

    def get_response_by_realization_id(self, response_definition_id, realization_id):
        return (
            self._session.query(Response)
            .filter_by(
                response_definition_id=response_definition_id,
                realization_id=realization_id)
            .one()
        )

    def get_realizations_by_response_name(self, response_name, ensemble_id):
        response_definition = (
            self._session.query(ResponseDefinition)
            .filter_by(name=response_name)
        )
        
        
        return (
            
        )