import pytest
from ert_shared.storage import ERT_STORAGE
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.storage_api import StorageApi


@pytest.fixture(scope="session")
def initialize_databases(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("database")
    url = f"sqlite:///{tmp_path}/ert_storage.db"
    ERT_STORAGE.initialize(url=url)


@pytest.fixture(scope="module")
def populated_database(initialize_databases):
    session = ERT_STORAGE.Session()
    api = RdbApi(session=session)

    db_lookup = {}
    prior_key1 = api.add_prior(
        "group", "key1", "function", ["paramA", "paramB"], [0.1, 0.2]
    )
    prior_key2 = api.add_prior(
        "group", "key2", "function", ["paramA", "paramB"], [0.3, 0.4]
    )
    prior_key3 = api.add_prior(
        "group", "key3", "function", ["paramA", "paramB"], [0.5, 0.6]
    )
    prior_a = api.add_prior("G", "A", "function", ["paramA", "paramB"], [0.5, 0.6])

    ######## add ensemble ########
    ensemble = api.add_ensemble(
        name="ensemble_name", priors=[prior_key1, prior_key2, prior_key3]
    )
    db_lookup["ensemble"] = ensemble.id
    db_lookup["ensemble_timestamp"] = ensemble.time_created
    ######## add parameteredefinitionss ########
    parameter_def_A_G = api.add_parameter_definition(
        "A", "G", "ensemble_name", prior=prior_a
    )
    parameter_def_B_G = api.add_parameter_definition("B", "G", "ensemble_name")
    parameter_def_key1_group = api.add_parameter_definition(
        "key1", "group", "ensemble_name", prior=prior_key1
    )
    db_lookup["parameter_def_A_G"] = parameter_def_A_G.id
    db_lookup["parameter_def_key1_group"] = parameter_def_key1_group.id

    ######## add observations ########
    observation_one = api.add_observation(
        name="observation_one",
        key_indices=[0, 3],
        data_indices=[2, 3],
        values=[10.1, 10.2],
        errors=[1, 3],
    )
    observation_one.add_attribute("region", "1")

    observation_two_first = api.add_observation(
        name="observation_two_first",
        key_indices=["2000-01-01 20:01:01"],
        data_indices=[4],
        values=[10.3],
        errors=[2],
    )

    observation_two_second = api.add_observation(
        name="observation_two_second",
        key_indices=["2000-01-02 20:01:01"],
        data_indices=[5],
        values=[10.4],
        errors=[2.5],
    )

    ######## add response definitions ########
    response_definition_one = api.add_response_definition(
        name="response_one",
        indices=[3, 5, 8, 9],
        ensemble_name=ensemble.name,
    )

    response_definition_two = api.add_response_definition(
        name="response_two",
        indices=[
            "2000-01-01 20:01:01",
            "2000-01-02 20:01:01",
            "2000-01-02 20:01:01",
            "2000-01-02 20:01:01",
            "2000-01-02 20:01:01",
            "2000-01-02 20:01:01",
        ],
        ensemble_name=ensemble.name,
    )
    db_lookup["response_definition_one"] = response_definition_one.id

    ######## observation response definition links ########
    obs_res_def_link = api._add_observation_response_definition_link(
        observation_id=observation_one.id,
        response_definition_id=response_definition_one.id,
        active=[True, False],
        update_id=None,
    )

    api._add_observation_response_definition_link(
        observation_id=observation_two_first.id,
        response_definition_id=response_definition_two.id,
        active=[True],
        update_id=None,
    )

    api._add_observation_response_definition_link(
        observation_id=observation_two_second.id,
        response_definition_id=response_definition_two.id,
        active=[True],
        update_id=None,
    )

    ######## add realizations ########
    realization_0 = api.add_realization(0, ensemble.name)
    realization_1 = api.add_realization(1, ensemble.name)
    db_lookup["realization_0"] = realization_0.id

    def add_data(realization, ens):
        response_one = api.add_response(
            name="response_one",
            values=[11.1, 11.2, 9.9, 9.3],
            realization_index=realization.index,
            ensemble_name=ens.name,
        )
        api._add_misfit(200, obs_res_def_link.id, response_one.id)

        api.add_response(
            name="response_two",
            values=[12.1, 12.2, 11.1, 11.2, 9.9, 9.3],
            realization_index=realization.index,
            ensemble_name=ens.name,
        )

        api.add_parameter("A", "G", 1, realization.index, "ensemble_name")
        api.add_parameter("B", "G", 2, realization.index, "ensemble_name")
        api.add_parameter("key1", "group", 2, realization.index, "ensemble_name")

    add_data(realization_0, ens=ensemble)
    add_data(realization_1, ens=ensemble)

    session.commit()
    session.close()

    yield db_lookup


@pytest.fixture
def api(initialize_databases):
    session = ERT_STORAGE.Session()
    api = RdbApi(session=session)

    try:
        yield api
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def db_api(populated_database):
    db_lookup = populated_database
    session = ERT_STORAGE.Session()
    api = RdbApi(session=session)

    try:
        yield api, db_lookup
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def storage_api(db_api):
    api, db_lookup = db_api
    yield StorageApi(session=ERT_STORAGE.Session()), db_lookup
