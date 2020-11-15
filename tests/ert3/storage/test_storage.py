import ert3

import pytest


def test_init(tmpdir):
    ert3.storage.init(tmpdir)


def test_double_init(tmpdir):
    ert3.storage.init(tmpdir)
    with pytest.raises(ValueError, match="Storage already initialized"):
        ert3.storage.init(tmpdir)


def test_double_add_experiment(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(tmpdir, "my_experiment")
    with pytest.raises(KeyError, match="Cannot initialize existing experiment"):
        ert3.storage.init_experiment(tmpdir, "my_experiment")


def test_add_experiments(tmpdir):
    ert3.storage.init(tmpdir)

    experiment_names = ["a", "b", "c", "super-experiment", "explosions"]
    for idx, experiment_name in enumerate(experiment_names):
        ert3.storage.init_experiment(tmpdir, experiment_name)
        expected_names = sorted(experiment_names[: idx + 1])
        retrieved_names = sorted(ert3.storage.get_experiment_names(tmpdir))
        assert expected_names == retrieved_names


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
    ert3.storage.init_experiment(tmpdir, "one")
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
    ert3.storage.init_experiment(tmpdir, "one")

    ert3.storage.add_input_data(tmpdir, "one", [])
    with pytest.raises(KeyError, match="Input data is already stored for experiment"):
        ert3.storage.add_input_data(tmpdir, "one", [])


def test_get_input_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(tmpdir, "one")

    input_data = {"a": "blabla", "c": [1, 2, 3]}
    ert3.storage.add_input_data(tmpdir, "one", input_data)

    retrieved_input_data = ert3.storage.get_input_data(tmpdir, "one")
    _assert_equal_data(input_data, retrieved_input_data)


def test_get_input_data_multiple_experiments(tmpdir):
    ert3.storage.init(tmpdir)

    ert3.storage.init_experiment(tmpdir, "one")
    input_data_one = {"a": "blabla", "c": [1, 2, 3]}
    ert3.storage.add_input_data(tmpdir, "one", input_data_one)

    ert3.storage.init_experiment(tmpdir, "two")
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
    ert3.storage.init_experiment(tmpdir, experiment_name)
    with pytest.raises(
        KeyError, match=f"No input data for experiment: {experiment_name}"
    ):
        ert3.storage.get_input_data(tmpdir, experiment_name)


def test_add_output_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(tmpdir, "one")
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
    ert3.storage.init_experiment(tmpdir, "one")

    ert3.storage.add_input_data(tmpdir, "one", [])
    ert3.storage.add_output_data(tmpdir, "one", [0, 1, 2, 3])
    with pytest.raises(KeyError, match="Output data is already stored for experiment"):
        ert3.storage.add_output_data(tmpdir, "one", [0, 1, 2, 3])


def test_add_output_data_no_input_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(tmpdir, "one")
    with pytest.raises(
        KeyError, match="Cannot add output data to experiment without input data"
    ):
        ert3.storage.add_output_data(tmpdir, "one", [0, 1, 2, 3])


def test_get_output_data(tmpdir):
    ert3.storage.init(tmpdir)
    ert3.storage.init_experiment(tmpdir, "one")

    ert3.storage.add_input_data(tmpdir, "one", [])

    output_data = {"a": "blabla", "c": [1, 2, 3]}
    ert3.storage.add_output_data(tmpdir, "one", output_data)
    retrieved_output_data = ert3.storage.get_output_data(tmpdir, "one")
    _assert_equal_data(output_data, retrieved_output_data)


def test_get_output_data_multiple_experiments(tmpdir):
    ert3.storage.init(tmpdir)

    ert3.storage.init_experiment(tmpdir, "one")
    output_data_one = {"a": "blabla", "c": [1, 2, 3]}
    ert3.storage.add_input_data(tmpdir, "one", [])
    ert3.storage.add_output_data(tmpdir, "one", output_data_one)

    ert3.storage.init_experiment(tmpdir, "two")
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
    ert3.storage.init_experiment(tmpdir, experiment_name)
    with pytest.raises(
        KeyError, match=f"No output data for experiment: {experiment_name}"
    ):
        ert3.storage.get_output_data(tmpdir, experiment_name)
