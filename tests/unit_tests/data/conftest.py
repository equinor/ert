from unittest.mock import Mock

import pytest


@pytest.fixture()
def facade():
    obs_mock = Mock()
    obs_mock.data_key = "test_data_key"

    facade = Mock()
    facade.get_impl.return_value = Mock()
    facade.get_ensemble_size.return_value = 3
    facade.get_observations.return_value = {"some_key": obs_mock}
    facade.get_data_key_for_obs_key.return_value = "some_key"

    return facade
