import time

import pandas as pd
import pytest
import sqlalchemy.exc
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.entities_model import Observation
from ert_shared.storage.rdb_api import RdbApi
from tests.storage import apis, db_apis, populated_database, initialize_databases


def test_add_observation(apis):
    rdb_api, blob_api = apis

    observation_name = "test"
    key_indexes = [0, 3]
    data_indexes = [0, 3]
    values = [22.1, 44.2]
    stds = [1, 3]
    key_indexes_df = blob_api.add_blob(data=key_indexes)
    data_indexes_df = blob_api.add_blob(data=data_indexes)
    values_df = blob_api.add_blob(data=values)
    stds_df = blob_api.add_blob(data=stds)

    observation = rdb_api.add_observation(
        name=observation_name,
        key_indexes_ref=key_indexes_df.id,
        data_indexes_ref=data_indexes_df.id,
        values_ref=values_df.id,
        stds_ref=stds_df.id,
    )
    assert observation is not None

    observation = rdb_api.get_observation(observation_name)
    assert observation is not None
    assert blob_api.get_blob(observation.key_indexes_ref).data == key_indexes
    assert blob_api.get_blob(observation.data_indexes_ref).data == data_indexes
    assert blob_api.get_blob(observation.values_ref).data == values
    assert blob_api.get_blob(observation.stds_ref).data == stds


def test_add_duplicate_observation(apis):
    rdb_api, blob_api = apis
    rdb_api.add_observation(
        name="test",
        key_indexes_ref=1,
        data_indexes_ref=1,
        values_ref=1,
        stds_ref=1,
    )

    with pytest.raises(sqlalchemy.exc.IntegrityError) as error:
        rdb_api.add_observation(
            name="test",
            key_indexes_ref=2,
            data_indexes_ref=2,
            values_ref=2,
            stds_ref=2,
        )


def test_observation_attribute(apis):
    rdb_api, blob_api = apis
    obs = rdb_api.add_observation(
        name="test",
        key_indexes_ref=1,
        data_indexes_ref=1,
        values_ref=1,
        stds_ref=1,
    )

    obs.add_attribute("foo", "bar")

    assert obs.get_attribute("foo") == "bar"


def test_add_response(apis):
    rdb_api, blob_api = apis
    indexes = [0, 2]
    values = [22.1, 44.2]

    indexes_df = blob_api.add_blob(data=indexes)
    values_df = blob_api.add_blob(data=values)

    ensemble = rdb_api.add_ensemble(name="test")

    response_definition = rdb_api.add_response_definition(
        name="test", indexes_ref=indexes_df.id, ensemble_name=ensemble.name
    )

    realization = rdb_api.add_realization(0, ensemble.name)

    response = rdb_api.add_response(
        name=response_definition.name,
        values_ref=values_df.id,
        realization_index=realization.index,
        ensemble_name=ensemble.name,
    )

    ensemble = rdb_api.get_ensemble(name="test")
    assert ensemble.id is not None
    response_definition = rdb_api._get_response_definition(
        name="test", ensemble_id=ensemble.id
    )
    assert response_definition.id is not None
    assert response_definition.ensemble_id is not None
    assert blob_api.get_blob(id=response_definition.indexes_ref).data == indexes

    realization = rdb_api.get_realization(index=0, ensemble_name=ensemble.name)
    assert realization.id is not None
    assert realization.ensemble_id is not None

    response = rdb_api.get_response(
        response_definition.name, realization.index, ensemble.name
    )
    assert response.id is not None
    assert response.realization_id is not None
    assert response.response_definition_id is not None
    assert blob_api.get_blob(id=response.values_ref).data == values


def test_add_ensemble(apis):
    rdb_api, blob_api = apis
    ensemble = rdb_api.add_ensemble(name="test_ensemble")
    assert ensemble.id is not None


def test_two_ensembles_with_same_name(apis):
    rdb_api, blob_api = apis
    ensemble1 = rdb_api.add_ensemble(name="test_ensemble")
    assert ensemble1.id is not None

    time.sleep(1)

    ensemble2 = rdb_api.add_ensemble(name="test_ensemble")
    assert ensemble2.id is not None

    ensemble = rdb_api.get_ensemble("test_ensemble")

    assert ensemble.id == ensemble2.id


def test_add_reference_ensemble(apis):
    rdb_api, blob_api = apis
    reference_ensemble_name = "test_ensemble"
    ensemble = rdb_api.add_ensemble(name=reference_ensemble_name)

    result_ensemble = rdb_api.add_ensemble(
        name="result_ensemble", reference=(reference_ensemble_name, "es_mda")
    )
    assert result_ensemble.parent.ensemble_reference.name == reference_ensemble_name


def test_add_realization(apis):
    rdb_api, blob_api = apis
    ensemble = rdb_api.add_ensemble(name="test_ensemble")

    realizations = []
    for i in range(5):
        realization = rdb_api.add_realization(i, ensemble.name)
        realizations.append(realization)

    assert ensemble.id is not None
    for realization in realizations:
        assert realization.id is not None

    with pytest.raises(sqlalchemy.exc.IntegrityError) as error:
        rdb_api.add_realization(0, ensemble_name=ensemble.name)


def test_add_parameter(apis):
    rdb_api, blob_api = apis
    value = 22.1

    value_df = blob_api.add_blob(data=value)

    ensemble = rdb_api.add_ensemble(name="test")

    parameter_definition = rdb_api.add_parameter_definition(
        name="test_param", group="test_group", ensemble_name=ensemble.name
    )

    realization = rdb_api.add_realization(0, ensemble.name)

    parameter = rdb_api.add_parameter(
        name=parameter_definition.name,
        group=parameter_definition.group,
        value_ref=value_df.id,
        realization_index=realization.index,
        ensemble_name=ensemble.name,
    )

    ensemble = rdb_api.get_ensemble(name="test")
    assert ensemble.id is not None

    parameter_definition = rdb_api._get_parameter_definition(
        name="test_param", group="test_group", ensemble_id=ensemble.id
    )
    assert parameter_definition.id is not None
    assert parameter_definition.ensemble_id is not None

    realization = rdb_api.get_realization(index=0, ensemble_name=ensemble.name)
    assert realization.id is not None
    assert realization.ensemble_id is not None

    parameter = rdb_api.get_parameter(
        name="test_param",
        group="test_group",
        realization_index=0,
        ensemble_name=ensemble.name,
    )
    assert parameter.id is not None
    assert parameter.realization_id is not None
    assert parameter.parameter_definition_id is not None
    assert blob_api.get_blob(id=parameter.value_ref).data == value


def test_add_observation_response_definition_link(apis):
    rdb_api, blob_api = apis
    observation = rdb_api.add_observation(
        name="test",
        key_indexes_ref=None,
        data_indexes_ref=None,
        values_ref=None,
        stds_ref=None,
    )

    ensemble = rdb_api.add_ensemble(name="test_ensemble")

    response_definition = rdb_api.add_response_definition(
        name="test_response_definition", indexes_ref=0, ensemble_name=ensemble.name
    )

    link = rdb_api._add_observation_response_definition_link(
        observation_id=observation.id,
        response_definition_id=response_definition.id,
        active_ref=1,
        update_id=None,
    )

    assert link.id is not None
    assert link.observation_id == observation.id
    assert link.response_definition_id == response_definition.id


def test_add_mistfit(apis):
    rdb_api, blob_api = apis
    observation = rdb_api.add_observation(
        name="test",
        key_indexes_ref=None,
        data_indexes_ref=None,
        values_ref=None,
        stds_ref=None,
    )

    ensemble = rdb_api.add_ensemble(name="test")

    response_definition = rdb_api.add_response_definition(
        name="test", indexes_ref=None, ensemble_name=ensemble.name
    )

    realization = rdb_api.add_realization(0, ensemble.name)

    response = rdb_api.add_response(
        name=response_definition.name,
        values_ref=0,
        realization_index=realization.index,
        ensemble_name=ensemble.name,
    )

    link = rdb_api._add_observation_response_definition_link(
        observation_id=observation.id,
        response_definition_id=response_definition.id,
        active_ref=1,
        update_id=None,
    )

    misfit = rdb_api._add_misfit(1.0, link.id, response.id)

    assert misfit.id is not None
    assert misfit.response_id == response.id
    assert misfit.observation_response_definition_link_id == link.id
    assert misfit.observation_response_definition_link.observation_id == observation.id


def test_get_parameter_bundle(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    bundle = rdb_api.get_parameter_bundle(
        parameter_def_id=db_lookup["parameter_def_A_G"],
        ensemble_id=db_lookup["ensemble"],
    )
    assert bundle is not None
    assert bundle.name == "A"
    assert bundle.group == "G"
    assert len(bundle.parameters) == 2


def test_add_prior(apis):
    rdb_api, blob_api = apis
    prior = rdb_api.add_prior("group", "key", "function", ["paramA", "paramB"], [1, 2])
    assert prior.id is not None


def test_get_response_data(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    responses = rdb_api.get_response_data(
        name="response_one", ensemble_name="ensemble_name"
    )
    ids = [resp.values_ref for resp in responses]
    assert ids is not None
    assert ids == [18, 23]


def test_get_response_bundle(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    bundle = rdb_api.get_response_bundle(
        response_name="response_one", ensemble_id=db_lookup["ensemble"]
    )
    assert bundle.name == "response_one"
    assert len(bundle.responses) == 2


def test_get_parameter_by_realization_id(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    param = rdb_api.get_parameter_by_realization_id(
        parameter_definition_id=db_lookup["parameter_def_A_G"],
        realization_id=db_lookup["realization_0"],
    )
    assert param.realization_id == db_lookup["realization_0"]
    assert param.parameter_definition_id == db_lookup["parameter_def_A_G"]


def test_get_parameter_definitions_by_ensemble_id(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    param_def = rdb_api.get_parameter_definitions_by_ensemble_id(
        ensemble_id=db_lookup["ensemble"]
    )
    assert len(list(param_def)) == 3


def test_get_response_by_realization_id(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    response = rdb_api.get_response_by_realization_id(
        response_definition_id=db_lookup["response_defition_one"],
        realization_id=db_lookup["realization_0"],
    )
    assert response is not None


def test_get_response_definitions_by_ensemble_id(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    resp_defs = rdb_api.get_response_definitions_by_ensemble_id(
        ensemble_id=db_lookup["ensemble"]
    )
    assert len(list(resp_defs)) == 2


def test_get_ensemble_by_id(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    ens = rdb_api.get_ensemble_by_id(ensemble_id=db_lookup["ensemble"])
    assert ens.name == "ensemble_name"


def test_get_realizations_by_ensemble_id(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    reals = rdb_api.get_realizations_by_ensemble_id(ensemble_id=db_lookup["ensemble"])
    assert len(list(reals)) == 2


def test_get_all_ensembles(db_apis):
    rdb_api, _, _ = db_apis
    ens_list = rdb_api.get_all_ensembles()
    assert len(list(ens_list)) == 1
    assert ens_list[0].name == "ensemble_name"


def test_get_all_observation_keys(db_apis):
    rdb_api, _, _ = db_apis
    obs_keys = rdb_api.get_all_observation_keys()
    assert set(
        ["observation_one", "observation_two_first", "observation_two_second"]
    ) == set(obs_keys)


def test_get_observation_attribute(db_apis):
    rdb_api, _, _ = db_apis
    obs_attrib = rdb_api.get_observation_attribute(
        name="observation_one", attribute="region"
    )
    assert obs_attrib == "1"


def test_get_observation_attributes(db_apis):
    rdb_api, _, _ = db_apis
    obs_attribs = rdb_api.get_observation_attributes(name="observation_one")
    assert obs_attribs is not None
    assert len(obs_attribs) == 1


def test_get_observation(db_apis):
    rdb_api, _, _ = db_apis
    obs = rdb_api.get_observation(name="observation_one")
    assert obs is not None
    assert obs.name == "observation_one"


def test_get_parameter(db_apis):
    rdb_api, _, _ = db_apis
    param = rdb_api.get_parameter(
        name="A", group="G", realization_index=0, ensemble_name="ensemble_name"
    )
    assert param.parameter_definition.name == "A"
    assert param.parameter_definition.group == "G"
    assert param.realization.index == 0
