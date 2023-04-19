import pytest
from hypothesis import assume, given

from ert._c_wrappers.enkf import ErtConfig

from .config_dict_generator import config_generators, to_config_file


@given(config_generators())
def test_ensemble_config_errors_on_unknown_function_in_field(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_values:
        assume(len(config_values.field) > 0)

        silly_function_name = "NORMALIZE_EGGS"
        fieldlist = list(config_values.field[0])
        alteredfieldlist = []

        for val in fieldlist:
            if "INIT_TRANSFORM" in val:
                alteredfieldlist.append("INIT_TRANSFORM:" + silly_function_name)
            else:
                alteredfieldlist.append(val)

        mylist = []
        mylist.append(tuple(alteredfieldlist))

        config_values.field = mylist

        to_config_file(filename, config_values)
        with pytest.raises(
            expected_exception=ValueError,
            match=f"FIELD INIT_TRANSFORM:{silly_function_name} is an invalid function",
        ):
            ErtConfig.from_file(filename)
