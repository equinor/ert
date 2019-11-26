import pytest
import sys

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


@pytest.fixture()
def facade():
    obs_mock = Mock()
    obs_mock.getDataKey.return_value = "test_data_key"
    obs_mock.getStepList.return_value.asList.return_value = [1]

    facade = Mock()
    facade.get_impl.return_value = Mock()
    facade.get_ensemble_size.return_value = 3
    facade.get_observations.return_value = {"some_key": obs_mock}

    facade.get_current_case_name.return_value = "test_case"

    return facade
