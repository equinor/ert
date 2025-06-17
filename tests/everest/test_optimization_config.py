import pytest
from pydantic import ValidationError

from everest.config import EverestConfig


def test_optimization_config_options(copy_test_data_to_tmp):
    config = EverestConfig.load_file("mocked_test_case/mocked_test_case.yml")
    config_dict = config.to_dict()

    config_dict["optimization"]["options"] = [
        "max_iterations = 0",
        "merit_function el_bakry",
    ]
    config = EverestConfig.model_validate(config_dict)

    config_dict["optimization"]["options"] = [
        "max_iterations = 0",
        "search_method = 1",
        "merit_function el_bakry",
    ]
    with pytest.raises(
        ValidationError, match=r"Input should be 'value_based_line_search',"
    ):
        config = EverestConfig.model_validate(config_dict)

    config_dict["optimization"]["options"] = [
        "max_iterations = 0",
        "foo = xyz",
        "bar",
        "merit_function el_bakry",
    ]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`, `bar`"
    ):
        config = EverestConfig.model_validate(config_dict)
