import os
from textwrap import dedent

import pytest
from hypothesis import assume, given

from ert._c_wrappers.enkf import ErtConfig
from ert.parsing import ConfigValidationError

from .config_dict_generator import config_generators


@given(config_generators())
def test_ensemble_config_errors_on_unknown_function_in_field(
    tmp_path_factory, config_generator
):
    with config_generator(tmp_path_factory) as config_values:
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

        with pytest.raises(
            expected_exception=ValueError,
            match=f"FIELD INIT_TRANSFORM:{silly_function_name} is an invalid function",
        ):
            _ = ErtConfig.from_dict(
                config_values.to_config_dict("test.ert", os.getcwd())
            )


def test_that_empty_grid_file_raises(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 10
        FIELD foo bar
        GRID grid.GRDECL
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("grid.GRDECL", "w", encoding="utf-8") as fh:
            fh.writelines("")

        with pytest.raises(
            expected_exception=ConfigValidationError,
            match="did not contain dimensions",
        ):
            _ = ErtConfig.from_file("config.ert")
