import pandas as pd
import pytest
import sqlalchemy.exc

from ert_shared.storage import Observation
from ert_shared.storage.repository import ErtRepository

from tests.storage import db_session, engine, tables


def test_add_observation(db_session):
    with ErtRepository(db_session) as repository:
        observation = repository.add_observation(
            name="test",
            key_indexes=[0, 3],
            data_indexes=[0, 3],
            values=[22.1, 44.2],
            stds=[1, 3],
        )
        repository.commit()
        assert observation is not None


def test_add_duplicate_observation(db_session):
    with ErtRepository(db_session) as repository:
        observation = repository.add_observation(
            name="test",
            key_indexes=[0, 3],
            data_indexes=[0, 3],
            values=[22.1, 44.2],
            stds=[1, 3],
        )
        repository.commit()

        with pytest.raises(sqlalchemy.exc.IntegrityError) as error:
            observation = repository.add_observation(
                name="test",
                key_indexes=[0, 3],
                data_indexes=[0, 3],
                values=[22.1, 44.2],
                stds=[1, 3],
            )
            repository.commit()


def test_add_response(db_session):
    with ErtRepository(db_session) as repository:
        ensemble = repository.add_ensemble(name="test")

        response_definition = repository.add_response_definition(
            name="test", indexes=[0, 2], ensemble_name=ensemble.name
        )

        realization = repository.add_realization(0, ensemble.name)

        response = repository.add_response(
            name=response_definition.name,
            values=[22.1, 44.2],
            realization_index=realization.index,
            ensemble_name=ensemble.name,
        )
        repository.commit()
        assert ensemble.id is not None
        assert response_definition.id is not None
        assert response_definition.ensemble_id is not None
        assert realization.id is not None
        assert realization.ensemble_id is not None
        assert response.id is not None
        assert response.realization_id is not None
        assert response.response_definition_id is not None


def test_add_ensemble(db_session):
    with ErtRepository(db_session) as repository:
        ensemble = repository.add_ensemble(name="test_ensemble")
        repository.commit()
        assert ensemble.id is not None

        with pytest.raises(sqlalchemy.exc.IntegrityError) as error:
            repository.add_ensemble(name="test_ensemble")
            repository.commit()


def test_add_realization(db_session):
    with ErtRepository(db_session) as repository:
        ensemble = repository.add_ensemble(name="test_ensemble")

        realizations = []
        for i in range(5):
            realization = repository.add_realization(i, ensemble.name)
            realizations.append(realization)

        repository.commit()

        assert ensemble.id is not None
        for realization in realizations:
            assert realization.id is not None

    with pytest.raises(sqlalchemy.exc.IntegrityError) as error, ErtRepository(
        session=db_session
    ) as repository:
        repository.add_realization(0, ensemble_name=ensemble.name)
        repository.commit()


def test_add_parameter(db_session):
    with ErtRepository(db_session) as repository:
        ensemble = repository.add_ensemble(name="test")

        parameter_definition = repository.add_parameter_definition(
            name="test_param", group="test_group", ensemble_name=ensemble.name
        )

        realization = repository.add_realization(0, ensemble.name)

        parameter = repository.add_parameter(
            name=parameter_definition.name,
            group=parameter_definition.group,
            value=22.1,
            realization_index=realization.index,
            ensemble_name=ensemble.name,
        )
        repository.commit()

        assert ensemble.id is not None
        assert parameter_definition.id is not None
        assert parameter_definition.ensemble_id is not None
        assert realization.id is not None
        assert realization.ensemble_id is not None
        assert parameter.id is not None
        assert parameter.realization_id is not None
        assert parameter.parameter_definition_id is not None
