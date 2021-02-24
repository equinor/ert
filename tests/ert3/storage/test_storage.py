import collections

import ert3

import pytest


def test_init(tmpdir):
    ert3.storage.init(workspace=tmpdir)


def test_double_init(tmpdir):
    ert3.storage.init(workspace=tmpdir)
    with pytest.raises(ValueError, match="Storage already initialized"):
        ert3.storage.init(workspace=tmpdir)


def test_double_add_experiment(tmpdir):
    ert3.storage.init(workspace=tmpdir)
    ert3.storage.init_experiment(
        workspace=tmpdir, experiment_name="my_experiment", parameters=[]
    )
    with pytest.raises(KeyError, match="Cannot initialize existing experiment"):
        ert3.storage.init_experiment(
            workspace=tmpdir, experiment_name="my_experiment", parameters=[]
        )


def test_add_experiments(tmpdir):
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


def test_add_input_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])
    ert3.storage.add_input_data(tmpdir, "one", {"a": "bla", "c": [1, 2, 3]})


def test_add_input_data_not_initialised(tmpdir):
    ert3.storage.init(tmpdir)
    with pytest.raises(
        KeyError, match="Cannot add input data to non-existing experiment"
    ):
        ert3.storage.add_input_data(tmpdir, "one", {"a": "bla", "c": [1, 2, 3]})


def test_add_input_data_not_initialised_storage(tmpdir):
    with pytest.raises(ValueError, match="Storage is not initialized"):
        ert3.storage.add_input_data(tmpdir, "one", {"a": "bla", "c": [1, 2, 3]})


def test_add_input_data_twice(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])

    ert3.storage.add_input_data(tmpdir, "one", [])
    with pytest.raises(KeyError, match="Input data is already stored for experiment"):
        ert3.storage.add_input_data(tmpdir, "one", [])


def test_get_input_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])

    input_data = {"a": "blabla", "c": [1, 2, 3]}
    ert3.storage.add_input_data(tmpdir, "one", input_data)

    retrieved_input_data = ert3.storage.get_input_data(tmpdir, "one")
    _assert_equal_data(input_data, retrieved_input_data)


def test_get_input_data_multiple_experiments(tmpdir):
    ert3.storage.init(tmpdir)

    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])
    input_data_one = {"a": "blabla", "c": [1, 2, 3]}
    ert3.storage.add_input_data(tmpdir, "one", input_data_one)

    ert3.storage.init_experiment(workspace=tmpdir, name="two", parameters=[])
    input_data_two = {"x": 10, "y": {7: [1, 2, 3]}}
    ert3.storage.add_input_data(tmpdir, "two", input_data_two)

    retrieved_input_data_one = ert3.storage.get_input_data(tmpdir, "one")
    _assert_equal_data(input_data_one, retrieved_input_data_one)

    retrieved_input_data_two = ert3.storage.get_input_data(tmpdir, "two")
    _assert_equal_data(input_data_two, retrieved_input_data_two)


def test_get_input_data_not_initialised(tmpdir):
    ert3.storage.init(tmpdir)
    with pytest.raises(
        KeyError, match="Cannot get input data, no experiment named: one"
    ):
        ert3.storage.get_input_data(tmpdir, "one")


def test_get_input_data_not_initialised_storage(tmpdir):
    with pytest.raises(ValueError, match="Storage is not initialized"):
        ert3.storage.get_input_data(tmpdir, "one")


@pytest.mark.parametrize(
    "experiment_name",
    ["one", "my_experiment", "some-name"],
)
def test_get_input_data_no_input_data(tmpdir, experiment_name):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name=experiment_name, parameters=[])
    with pytest.raises(
        KeyError, match=f"No input data for experiment: {experiment_name}"
    ):
        ert3.storage.get_input_data(tmpdir, experiment_name)


def test_add_output_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])
    ert3.storage.add_input_data(tmpdir, "one", {"a": "bla", "c": [1, 2, 3]})
    ert3.storage.add_output_data(tmpdir, "one", [0, 1, 2, 3])


def test_add_output_data_not_initialised(tmpdir):
    ert3.storage.init(tmpdir)
    with pytest.raises(
        KeyError, match="Cannot add output data to non-existing experiment"
    ):
        ert3.storage.add_output_data(tmpdir, "one", {"a": "bla", "c": [1, 2, 3]})


def test_add_output_data_not_initialised_storage(tmpdir):
    with pytest.raises(ValueError, match="Storage is not initialized"):
        ert3.storage.add_output_data(tmpdir, "one", {"a": "bla", "c": [1, 2, 3]})


def test_add_output_data_twice(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])

    ert3.storage.add_input_data(tmpdir, "one", [])
    ert3.storage.add_output_data(tmpdir, "one", [0, 1, 2, 3])
    with pytest.raises(KeyError, match="Output data is already stored for experiment"):
        ert3.storage.add_output_data(tmpdir, "one", [0, 1, 2, 3])


def test_add_output_data_no_input_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])
    with pytest.raises(
        KeyError, match="Cannot add output data to experiment without input data"
    ):
        ert3.storage.add_output_data(tmpdir, "one", [0, 1, 2, 3])


def test_get_output_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])

    ert3.storage.add_input_data(tmpdir, "one", [])

    output_data = {"a": "blabla", "c": [1, 2, 3]}
    ert3.storage.add_output_data(tmpdir, "one", output_data)
    retrieved_output_data = ert3.storage.get_output_data(tmpdir, "one")
    _assert_equal_data(output_data, retrieved_output_data)


def test_get_output_data_multiple_experiments(tmpdir):
    ert3.storage.init(tmpdir)

    ert3.storage.init_experiment(workspace=tmpdir, name="one", parameters=[])
    output_data_one = {"a": "blabla", "c": [1, 2, 3]}
    ert3.storage.add_input_data(tmpdir, "one", [])
    ert3.storage.add_output_data(tmpdir, "one", output_data_one)

    ert3.storage.init_experiment(workspace=tmpdir, name="two", parameters=[])
    output_data_two = {"x": 10, "y": {7: [1, 2, 3]}}
    ert3.storage.add_input_data(tmpdir, "two", [])
    ert3.storage.add_output_data(tmpdir, "two", output_data_two)

    retrieved_output_data_one = ert3.storage.get_output_data(tmpdir, "one")
    _assert_equal_data(output_data_one, retrieved_output_data_one)

    retrieved_output_data_two = ert3.storage.get_output_data(tmpdir, "two")
    _assert_equal_data(output_data_two, retrieved_output_data_two)


def test_get_output_data_not_initialised(tmpdir):
    ert3.storage.init(tmpdir)
    with pytest.raises(
        KeyError, match="Cannot get output data, no experiment named: one"
    ):
        ert3.storage.get_output_data(tmpdir, "one")


def test_get_output_data_not_initialised_storage(tmpdir):
    with pytest.raises(ValueError, match="Storage is not initialized"):
        ert3.storage.get_output_data(tmpdir, "one")


@pytest.mark.parametrize(
    "experiment_name",
    ["one", "my_experiment", "some-name"],
)
def test_get_output_data_no_output_data(tmpdir, experiment_name):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(workspace=tmpdir, name=experiment_name, parameters=[])
    with pytest.raises(
        KeyError, match=f"No output data for experiment: {experiment_name}"
    ):
        ert3.storage.get_output_data(tmpdir, experiment_name)


def test_add_sample_not_initialised(tmpdir):
    with pytest.raises(ValueError, match="Storage is not initialized"):
        ert3.storage.add_variables(tmpdir, "some_sample", [1, 2, 3])


def test_add_sample_twice(tmpdir):
    ert3.storage.init(tmpdir)

    ert3.storage.add_variables(tmpdir, "some_sample", [1, 2, 3])
    with pytest.raises(KeyError):
        ert3.storage.add_variables(tmpdir, "some_sample", [1, 2, 3])


def test_add_and_get_samples(tmpdir):
    ert3.storage.init(tmpdir)

    sample1 = [1, 2, 3.0]
    ert3.storage.add_variables(tmpdir, "sample1", sample1)

    sample2 = {i * "key": 1.5 * i for i in range(1, 10)}
    ert3.storage.add_variables(tmpdir, "sample2", sample2)

    assert ert3.storage.get_variables(tmpdir, "sample1") == sample1
    assert ert3.storage.get_variables(tmpdir, "sample2") == sample2


@pytest.mark.parametrize(
    "raw_ensrec",
    (
        [{"data": [i + 0.5, i + 1.1, i + 2.2]} for i in range(3)],
        [{"data": {"a": i + 0.5, "b": i + 1.1, "c": i + 2.2}} for i in range(5)],
        [{"data": {2: i + 0.5, 5: i + 1.1, 7: i + 2.2}} for i in range(2)],
    ),
)
def test_add_and_get_ensemble_record(tmpdir, raw_ensrec):
    ert3.storage.init(workspace=tmpdir)

    ensrecord = ert3.data.EnsembleRecord(records=raw_ensrec)
    ert3.storage.add_ensemble_record(
        workspace=tmpdir, record_name="my_ensemble_record", ensemble_record=ensrecord
    )
    retrieved_ensrecord = ert3.storage.get_ensemble_record(
        workspace=tmpdir, record_name="my_ensemble_record"
    )

    assert ensrecord == retrieved_ensrecord


def test_add_and_get_experiment_ensemble_record(tmpdir):
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


def test_get_record_names(tmpdir):
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
