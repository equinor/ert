import pytest
from ert_shared.storage import database_schema as ds
from ert_shared.storage.app import app
from ert_shared.storage.db import ErtStorage, get_db
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def db_factory(tmp_path_factory):
    """Database-creating factory. Note that ideally this factory should be created
    with static test-data that always exist, and it should roll-back every test.
    This doesn't work at the moment, and so care should be taken to make sure
    that the tests don't interfere with each other.
    """

    tmp_path = tmp_path_factory.mktemp("database")

    sess = ErtStorage()
    sess.initialize(project_path=tmp_path, testing=True)

    def override_get_db():
        try:
            db = sess.Session()
            yield db
            db.commit()
            db.flush()
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield sess.Session


@pytest.fixture(scope="module")
def db_populated(db_factory):
    db = db_factory()

    priors = [
        ds.ParameterPrior(
            group="group",
            key="key1",
            function="function",
            parameter_names=["paramA", "paramB"],
            parameter_values=[0.1, 0, 2],
        ),
        ds.ParameterPrior(
            group="group",
            key="key2",
            function="function",
            parameter_names=["paramA", "paramB"],
            parameter_values=[0.3, 0, 4],
        ),
        ds.ParameterPrior(
            group="group",
            key="key3",
            function="function",
            parameter_names=["paramA", "paramB"],
            parameter_values=[0.5, 0, 6],
        ),
    ]
    db.add_all(priors)

    prior_a = ds.ParameterPrior(
        group="G",
        key="A",
        function="function",
        parameter_names=["paramA", "paramB"],
        parameter_values=[0.5, 0.6],
    )
    db.add(prior_a)

    ######## add ensemble ########
    ensemble = ds.Ensemble(name="ensemble_name", priors=priors, num_realizations=2)
    db.add(ensemble)

    ######## add parameteredefinitionss ########
    parameter_def_A_G = ds.Parameter(
        name="A",
        group="G",
        ensemble=ensemble,
        prior=prior_a,
        values=[1, 1],
    )
    parameter_def_B_G = ds.Parameter(
        name="B",
        group="G",
        ensemble=ensemble,
        values=[2, 2],
    )
    parameter_def_key1_group = ds.Parameter(
        name="key1",
        group="group",
        ensemble=ensemble,
        prior=priors[0],
        values=[2, 2],
    )
    db.add_all(
        [
            parameter_def_A_G,
            parameter_def_B_G,
            parameter_def_key1_group,
        ]
    )

    ######## add observations ########
    observation_one = ds.Observation(
        name="observation_one",
        x_axis=[0, 3],
        values=[10.1, 10.2],
        errors=[1, 3],
        attributes={"region": ds.AttributeValue("1")},
    )

    observation_two_first = ds.Observation(
        name="observation_two_first",
        x_axis=["2000-01-01T20:01:01Z"],
        values=[10.3],
        errors=[2],
    )

    observation_two_second = ds.Observation(
        name="observation_two_second",
        x_axis=["2000-01-02T20:01:01Z"],
        values=[10.4],
        errors=[2.5],
    )
    db.add_all(
        (
            observation_one,
            observation_two_first,
            observation_two_second,
        )
    )

    ######## add response definitions ########
    response_definition_one = ds.ResponseDefinition(
        name="response_one",
        indices=[0, 3, 5, 8, 9],
        ensemble=ensemble,
    )

    response_definition_two = ds.ResponseDefinition(
        name="response_two",
        indices=[
            "2000-01-01T20:01:01Z",
            "2000-01-02T20:01:01Z",
            "2000-01-02T20:01:01Z",
            "2000-01-02T20:01:01Z",
            "2000-01-02T20:01:01Z",
            "2000-01-02T20:01:01Z",
        ],
        ensemble=ensemble,
    )
    db.add_all(
        (
            response_definition_one,
            response_definition_two,
        )
    )

    ######## observation response definition links ########
    obs_res_def_link = ds.ObservationResponseDefinitionLink(
        observation=observation_one,
        response_definition=response_definition_one,
        active=[True, False],
        update_id=None,
    )

    db.add_all(
        [
            obs_res_def_link,
            ds.ObservationResponseDefinitionLink(
                observation=observation_two_first,
                response_definition=response_definition_two,
                active=[True],
                update_id=None,
            ),
            ds.ObservationResponseDefinitionLink(
                observation=observation_two_second,
                response_definition=response_definition_two,
                active=[True],
                update_id=None,
            ),
        ]
    )

    ######## add responses ########
    def add_data(index):
        response_one = ds.Response(
            response_definition=response_definition_one,
            values=[10.2, 11.1, 11.2, 9.9, 9.3],
            index=index,
        )
        db.add(response_one)
        db.add(
            ds.Misfit(
                value=200,
                observation_response_definition_link=obs_res_def_link,
                response=response_one,
            )
        )

        db.add(
            ds.Response(
                response_definition=response_definition_two,
                values=[12.1, 12.2, 11.1, 11.2, 9.9, 9.3],
                index=index,
            )
        )

    add_data(0)
    add_data(1)

    db.commit()
    db.close()

    yield db_factory


@pytest.fixture
def db(db_populated):
    try:
        db = db_populated()
        yield db
    finally:
        db.close()


@pytest.fixture
def app_client(db):
    tc = TestClient(app)
    tc.db = db
    yield tc
