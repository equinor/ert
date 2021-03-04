import collections
from ert3 import workspace

import ert3

import pytest


@pytest.mark.requires_ert_storage
def test_init(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)


@pytest.mark.requires_ert_storage
def test_double_init(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    with pytest.raises(ValueError, match="Storage already initialized"):
        ert3.storage.init(workspace=tmpdir)


@pytest.mark.requires_ert_storage
def test_double_add_experiment(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    ert3.storage.init_experiment(
        workspace=tmpdir, experiment_name="my_experiment", parameters=[]
    )
    with pytest.raises(KeyError, match="Cannot initialize existing experiment"):
        ert3.storage.init_experiment(
            workspace=tmpdir, experiment_name="my_experiment", parameters=[]
        )


@pytest.mark.requires_ert_storage
def test_add_experiments(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    experiment_names = ["a", "b", "c", "super-experiment", "explosions"]
    experiment_parameters = [
        ["x"],
        ["a", "b"],
        ["alpha", "beta"],
        ["oxygen", "heat", "fuel"],
    ]
    experiments = zip(experiment_names, experiment_parameters)
    for idx, (experiment_name, experiment_parameters) in enumerate(experiments):
        ert3.storage.init_experiment(
            workspace=tmpdir,
            experiment_name=experiment_name,
            parameters=experiment_parameters,
        )
        expected_names = sorted(experiment_names[: idx + 1])
        retrieved_names = sorted(ert3.storage.get_experiment_names(workspace=tmpdir))
        assert expected_names == retrieved_names

        parameters = ert3.storage.get_experiment_parameters(
            workspace=tmpdir, experiment_name=experiment_name
        )
        assert experiment_parameters == parameters


def _assert_equal_data(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert sorted(a.keys()) == sorted(b.keys())
        for key in a.keys():
            _assert_equal_data(a[key], b[key])
    elif isinstance(a, list):
        assert isinstance(b, list)
        assert len(a) == len(b)
        for elem_a, elem_b in zip(a, b):
            _assert_equal_data(elem_a, elem_b)
    else:
        assert a == b


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize(
    "raw_ensrec",
    (
        [{"data": [i + 0.5, i + 1.1, i + 2.2]} for i in range(3)],
        [{"data": {"a": i + 0.5, "b": i + 1.1, "c": i + 2.2}} for i in range(5)],
        [{"data": {2: i + 0.5, 5: i + 1.1, 7: i + 2.2}} for i in range(2)],
    ),
)
def test_add_and_get_ensemble_record(tmpdir, raw_ensrec, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    ensrecord = ert3.data.EnsembleRecord(records=raw_ensrec)
    ert3.storage.add_ensemble_record(
        workspace=tmpdir, record_name="my_ensemble_record", ensemble_record=ensrecord
    )
    retrieved_ensrecord = ert3.storage.get_ensemble_record(
        workspace=tmpdir, record_name="my_ensemble_record"
    )

    assert ensrecord == retrieved_ensrecord


@pytest.mark.requires_ert_storage
def test_add_and_get_experiment_ensemble_record(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    for eid in range(1, 2):
        experiment = eid * "e"
        ert3.storage.init_experiment(
            workspace=tmpdir, experiment_name=experiment, parameters=[]
        )
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert3.data.EnsembleRecord(
                records=[
                    ert3.data.Record(data=[nid * eid * rid]) for rid in range(0, 5)
                ]
            )
            ert3.storage.add_ensemble_record(
                workspace=tmpdir,
                record_name=name,
                ensemble_record=ensemble_record,
                experiment_name=experiment,
            )

    for eid in range(1, 2):
        experiment = eid * "e"
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert3.data.EnsembleRecord(
                records=[
                    ert3.data.Record(data=[nid * eid * rid]) for rid in range(0, 5)
                ]
            )
            fetched_ensemble_record = ert3.storage.get_ensemble_record(
                workspace=tmpdir, record_name=name, experiment_name=experiment
            )
            assert ensemble_record == fetched_ensemble_record


@pytest.mark.requires_ert_storage
def test_get_record_names(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    experiment_records = collections.defaultdict(list)
    for eid in [None, 1, 2, 3]:
        experiment = eid if eid is None else "e" + str(eid)
        ert3.storage.init_experiment(
            workspace=tmpdir, experiment_name=experiment, parameters=[]
        )
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert3.data.EnsembleRecord(
                records=[ert3.data.Record(data=[0]) for rid in range(0, 5)]
            )
            ert3.storage.add_ensemble_record(
                workspace=tmpdir,
                record_name=name,
                ensemble_record=ensemble_record,
                experiment_name=experiment,
            )
            experiment_records[str(experiment)].append(name)

            recnames = ert3.storage.get_ensemble_record_names(
                workspace=tmpdir, experiment_name=experiment
            )
            assert sorted(experiment_records[str(experiment)]) == sorted(recnames)


@pytest.mark.requires_ert_storage
def test_delete_experiment(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    ert3.storage.init_experiment(
        workspace=tmpdir, experiment_name="test", parameters=[]
    )

    assert "test" in ert3.storage.get_experiment_names(workspace=tmpdir)

    with pytest.raises(
        ert3.exceptions.NonExistantExperiment,
        match="Experiment does not exist: does_not_exist",
    ):
        ert3.storage.delete_experiment(
            workspace=tmpdir, experiment_name="does_not_exist"
        )

    ert3.storage.delete_experiment(workspace=tmpdir, experiment_name="test")

    assert "test" not in ert3.storage.get_experiment_names(workspace=tmpdir)
