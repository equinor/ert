import time
import pytest
import sqlalchemy.exc
from tests.storage import initialize_databases, api, populated_database, db_api


def test_add_observation(api):

    observation_name = "test"
    key_indexes = [0, 3]
    data_indexes = [0, 3]
    values = [22.1, 44.2]
    stds = [1, 3]

    observation = api.add_observation(
        name=observation_name,
        key_indices=key_indexes,
        data_indices=data_indexes,
        values=values,
        errors=stds,
    )
    assert observation is not None

    observation = api.get_observation(observation_name)
    assert observation is not None
    assert observation.key_indices == key_indexes
    assert observation.data_indices == data_indexes
    assert observation.values == values
    assert observation.errors == stds


def test_add_duplicate_observation(api):
    api.add_observation(
        name="test",
        key_indices=1,
        data_indices=1,
        values=1,
        errors=1,
    )

    with pytest.raises(sqlalchemy.exc.IntegrityError) as error:
        api.add_observation(
            name="test",
            key_indices=2,
            data_indices=2,
            values=2,
            errors=2,
        )


def test_observation_attribute(api):
    obs = api.add_observation(
        name="test",
        key_indices=1,
        data_indices=1,
        values=1,
        errors=1,
    )

    obs.add_attribute("foo", "bar")

    assert obs.get_attribute("foo") == "bar"


def test_add_response(api):
    indices = [0, 2]
    values = [22.1, 44.2]

    ensemble = api.add_ensemble(name="test")

    response_definition = api.add_response_definition(
        name="test", indices=indices, ensemble_name=ensemble.name
    )

    realization = api.add_realization(0, ensemble.name)

    _ = api.add_response(
        name=response_definition.name,
        values=values,
        realization_index=realization.index,
        ensemble_name=ensemble.name,
    )

    ensemble = api.get_ensemble(name="test")
    assert ensemble.id is not None
    response_definition = api._get_response_definition(
        name="test", ensemble_id=ensemble.id
    )
    assert response_definition.id is not None
    assert response_definition.ensemble_id is not None
    assert response_definition.indices == indices

    realization = api.get_realization(index=0, ensemble_name=ensemble.name)
    assert realization.id is not None
    assert realization.ensemble_id is not None

    response = api.get_response(
        response_definition.name, realization.index, ensemble.name
    )
    assert response.id is not None
    assert response.realization_id is not None
    assert response.response_definition_id is not None
    assert response.values == values


def test_add_ensemble(api):
    ensemble = api.add_ensemble(name="test_ensemble")
    assert ensemble.id is not None


def test_two_ensembles_with_same_name(api):
    ensemble1 = api.add_ensemble(name="test_ensemble")
    assert ensemble1.id is not None

    time.sleep(1)

    ensemble2 = api.add_ensemble(name="test_ensemble")
    assert ensemble2.id is not None

    ensemble = api.get_ensemble("test_ensemble")

    assert ensemble.id == ensemble2.id


def test_add_reference_ensemble(api):
    reference_ensemble_name = "test_ensemble"
    ensemble = api.add_ensemble(name=reference_ensemble_name)

    result_ensemble = api.add_ensemble(
        name="result_ensemble", reference=(reference_ensemble_name, "es_mda")
    )
    assert result_ensemble.parent.ensemble_reference.name == reference_ensemble_name


def test_add_realization(api):
    ensemble = api.add_ensemble(name="test_ensemble")

    realizations = []
    for i in range(5):
        realization = api.add_realization(i, ensemble.name)
        realizations.append(realization)

    assert ensemble.id is not None
    for realization in realizations:
        assert realization.id is not None

    with pytest.raises(sqlalchemy.exc.IntegrityError) as error:
        api.add_realization(0, ensemble_name=ensemble.name)


def test_add_parameter(api):
    value = 22.1

    ensemble = api.add_ensemble(name="test")
    parameter_definition = api.add_parameter_definition(
        name="test_param", group="test_group", ensemble_name=ensemble.name
    )

    realization = api.add_realization(0, ensemble.name)

    parameter = api.add_parameter(
        name=parameter_definition.name,
        group=parameter_definition.group,
        value=value,
        realization_index=realization.index,
        ensemble_name=ensemble.name,
    )

    ensemble = api.get_ensemble(name="test")
    assert ensemble.id is not None

    parameter_definition = api._get_parameter_definition(
        name="test_param", group="test_group", ensemble_id=ensemble.id
    )
    assert parameter_definition.id is not None
    assert parameter_definition.ensemble_id is not None

    realization = api.get_realization(index=0, ensemble_name=ensemble.name)
    assert realization.id is not None
    assert realization.ensemble_id is not None

    parameter = api.get_parameter(
        name="test_param",
        group="test_group",
        realization_index=0,
        ensemble_name=ensemble.name,
    )
    assert parameter.id is not None
    assert parameter.realization_id is not None
    assert parameter.parameter_definition_id is not None
    assert parameter.value == value


def test_add_observation_response_definition_link(api):
    observation = api.add_observation(
        name="test",
        key_indices=None,
        data_indices=None,
        values=None,
        errors=None,
    )

    ensemble = api.add_ensemble(name="test_ensemble")

    response_definition = api.add_response_definition(
        name="test_response_definition", indices=0, ensemble_name=ensemble.name
    )

    link = api._add_observation_response_definition_link(
        observation_id=observation.id,
        response_definition_id=response_definition.id,
        active=1,
        update_id=None,
    )

    assert link.id is not None
    assert link.observation_id == observation.id
    assert link.response_definition_id == response_definition.id


def test_add_misfits(api):
    observation = api.add_observation(
        name="test",
        key_indices=None,
        data_indices=None,
        values=None,
        errors=None,
    )

    ensemble = api.add_ensemble(name="test")

    response_definition = api.add_response_definition(
        name="test", indices=None, ensemble_name=ensemble.name
    )

    realization = api.add_realization(0, ensemble.name)

    response = api.add_response(
        name=response_definition.name,
        values=0,
        realization_index=realization.index,
        ensemble_name=ensemble.name,
    )

    link = api._add_observation_response_definition_link(
        observation_id=observation.id,
        response_definition_id=response_definition.id,
        active=1,
        update_id=None,
    )

    misfit = api._add_misfit(1.0, link.id, response.id)

    assert misfit.id is not None
    assert misfit.response_id == response.id
    assert misfit.observation_response_definition_link_id == link.id
    assert misfit.observation_response_definition_link.observation_id == observation.id


def test_get_parameter_bundle(db_api):
    api, db_lookup = db_api
    bundle = api.get_parameter_bundle(
        parameter_def_id=db_lookup["parameter_def_A_G"],
        ensemble_id=db_lookup["ensemble"],
    )
    assert bundle is not None
    assert bundle.name == "A"
    assert bundle.group == "G"
    assert len(bundle.parameters) == 2


def test_add_prior(api):
    prior = api.add_prior("group", "key", "function", ["paramA", "paramB"], [1, 2])
    assert prior.id is not None


def test_get_response_data(db_api):
    api, db_lookup = db_api
    responses = api.get_response_data(
        name="response_one", ensemble_name="ensemble_name"
    )
    vals = [resp.values for resp in responses]
    assert vals is not None
    assert vals == [[11.1, 11.2, 9.9, 9.3], [11.1, 11.2, 9.9, 9.3]]


def test_get_response_bundle(db_api):
    api, db_lookup = db_api
    bundle = api.get_response_bundle(
        response_name="response_one", ensemble_id=db_lookup["ensemble"]
    )
    assert bundle.name == "response_one"
    assert len(bundle.responses) == 2


def test_get_parameter_by_realization_id(db_api):
    api, db_lookup = db_api
    param = api.get_parameter_by_realization_id(
        parameter_definition_id=db_lookup["parameter_def_A_G"],
        realization_id=db_lookup["realization_0"],
    )
    assert param.realization_id == db_lookup["realization_0"]
    assert param.parameter_definition_id == db_lookup["parameter_def_A_G"]


def test_get_parameter_definitions_by_ensemble_id(db_api):
    api, db_lookup = db_api
    param_def = api.get_parameter_definitions_by_ensemble_id(
        ensemble_id=db_lookup["ensemble"]
    )
    assert len(list(param_def)) == 3


def test_get_response_by_realization_id(db_api):
    api, db_lookup = db_api
    response = api.get_response_by_realization_id(
        response_definition_id=db_lookup["response_definition_one"],
        realization_id=db_lookup["realization_0"],
    )
    assert response is not None


def test_get_response_definitions_by_ensemble_id(db_api):
    api, db_lookup = db_api
    resp_defs = api.get_response_definitions_by_ensemble_id(
        ensemble_id=db_lookup["ensemble"]
    )
    assert len(list(resp_defs)) == 2


def test_get_ensemble_by_id(db_api):
    api, db_lookup = db_api
    ens = api.get_ensemble_by_id(ensemble_id=db_lookup["ensemble"])
    assert ens.name == "ensemble_name"


def test_get_realizations_by_ensemble_id(db_api):
    api, db_lookup = db_api
    reals = api.get_realizations_by_ensemble_id(ensemble_id=db_lookup["ensemble"])
    assert len(list(reals)) == 2


def test_get_all_ensembles(db_api):
    api, _ = db_api
    ens_list = api.get_all_ensembles()
    assert len(list(ens_list)) == 1
    assert ens_list[0].name == "ensemble_name"


def test_get_all_observation_keys(db_api):
    api, _ = db_api
    obs_keys = api.get_all_observation_keys()
    assert set(
        ["observation_one", "observation_two_first", "observation_two_second"]
    ) == set(obs_keys)


def test_get_observation_attribute(db_api):
    api, _ = db_api
    obs_attrib = api.get_observation_attribute(
        name="observation_one", attribute="region"
    )
    assert obs_attrib == "1"


def test_get_observation_attributes(db_api):
    api, _ = db_api
    obs_attribs = api.get_observation_attributes(name="observation_one")
    assert obs_attribs is not None
    assert len(obs_attribs) == 1


def test_get_observation(db_api):
    api, _ = db_api
    obs = api.get_observation(name="observation_one")
    assert obs is not None
    assert obs.name == "observation_one"


def test_get_parameter(db_api):
    api, _ = db_api
    param = api.get_parameter(
        name="A", group="G", realization_index=0, ensemble_name="ensemble_name"
    )
    assert param.parameter_definition.name == "A"
    assert param.parameter_definition.group == "G"
    assert param.realization.index == 0
