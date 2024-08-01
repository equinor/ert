import os
from pathlib import Path

import pytest
import xtgeo

from ert.config import ConfigValidationError, ConfigWarning, Field
from ert.config.field import TRANSFORM_FUNCTIONS
from ert.config.parsing import init_user_config_schema, lark_parse
from ert.enkf_main import sample_prior
from ert.field_utils import Shape, read_field


def test_write_to_runpath_produces_the_transformed_field_in_storage(
    snake_oil_field_example, storage
):
    ensemble_config = snake_oil_field_example.ensemble_config
    experiment_id = storage.create_experiment(
        parameters=ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=5
    )
    active_realizations = [0, 3, 4]
    sample_prior(prior_ensemble, active_realizations)
    permx_field = ensemble_config["PERMX"]
    assert (permx_field.nx, permx_field.ny, permx_field.nz) == (10, 10, 5)
    assert permx_field.truncation_min is None
    assert permx_field.truncation_max is None
    assert permx_field.input_transformation is None
    assert permx_field.output_transformation is None

    for real in active_realizations:
        permx_field.write_to_runpath(
            Path(f"export/with/path/{real}"), real, prior_ensemble
        )
        assert read_field(
            f"export/with/path/{real}/permx.grdecl",
            "PERMX",
            permx_field.mask,
            Shape(permx_field.nx, permx_field.ny, permx_field.nz),
        ).flatten().tolist() == pytest.approx(
            permx_field._transform_data(
                permx_field._fetch_from_ensemble(real, prior_ensemble)
            )
            .flatten()
            .tolist()
        )
    for real in [1, 2]:
        with pytest.raises(
            KeyError, match=f"No dataset 'PERMX' in storage for realization {real}"
        ):
            permx_field.write_to_runpath(
                Path(f"export/with/path/{real}"), real, prior_ensemble
            )
        assert not os.path.isfile(f"export/with/path/{real}/permx.grdecl")


@pytest.fixture
def grid_shape():
    return Shape(2, 3, 4)


@pytest.fixture
def egrid_file(tmp_path, grid_shape):
    grid = xtgeo.create_box_grid(
        dimension=(grid_shape.nx, grid_shape.ny, grid_shape.nz)
    )
    grid.to_file(tmp_path / "MY_EGRID.EGRID", "egrid")
    return str(tmp_path / "MY_EGRID.EGRID")


@pytest.fixture
def parse_field_line(tmp_path, grid_shape, egrid_file):
    config_file = tmp_path / "test.ert"

    def make_field(field_line):
        config_file.write_text(
            "NUM_REALIZATIONS 1\nGRID " + egrid_file + "\n" + field_line + "\n",
            encoding="utf-8",
        )
        parsed = lark_parse(str(config_file), init_user_config_schema(), None, None)

        return Field.from_config_list(parsed["GRID"], grid_shape, parsed["FIELD"][0])

    return make_field


def test_field_dimensions_are_gotten_from_the_grid(parse_field_line, grid_shape):
    field = parse_field_line(
        "FIELD PERMX PARAMETER permx.grdecl INIT_FILES:fields/perms%d.grdecl"
    )
    assert (field.nx, field.ny, field.nz) == tuple(grid_shape)


@pytest.mark.parametrize(
    "ext",
    [
        "roff_binary",
        "roff_ascii",
        "roff",
        "grdecl",
        "bgrdecl",
        "ROFF_BINARY",
        "ROFF_ASCII",
        "ROFF",
        "GRDECL",
        "BGRDECL",
    ],
)
def test_file_format_is_gotten_from_the_output_file(parse_field_line, ext):
    field = parse_field_line(f"FIELD f PARAMETER f.{ext} INIT_FILES:f%d.grdecl")
    assert field.file_format.name == ext.upper()
    assert field.output_file == Path(f"f.{ext}")


@pytest.mark.parametrize(
    "ext, expected",
    [
        (
            "",
            r"Line 3.*Missing extension for field output file 'param',"
            r".*'ROFF_BINARY', 'ROFF_ASCII', 'ROFF', 'GRDECL', 'BGRDECL'",
        ),
        (".wrong", r"Line 3.*Unknown file format for output file: '.wrong'"),
    ],
)
def test_unknown_file_extensions_raises_config_validation_error(
    ext, expected, parse_field_line
):
    with pytest.raises(ConfigValidationError, match=expected):
        _ = parse_field_line(f"FIELD F PARAMETER param{ext} INIT_FILES:f.grdecl")


@pytest.mark.parametrize("transform", TRANSFORM_FUNCTIONS)
def test_output_transform_is_gotten_from_keyword(parse_field_line, transform):
    field = parse_field_line(
        f"FIELD f PARAMETER f.roff INIT_FILES:f%d.grdecl OUTPUT_TRANSFORM:{transform}"
    )
    assert field.output_transformation == transform


@pytest.mark.parametrize("transform", TRANSFORM_FUNCTIONS)
def test_init_transform_is_gotten_from_keyword(parse_field_line, transform):
    field = parse_field_line(
        f"FIELD f PARAMETER f.roff INIT_FILES:f%d.grdecl INIT_TRANSFORM:{transform}"
    )
    assert field.input_transformation == transform


@pytest.mark.parametrize("transform", ["INIT_TRANSFORM", "OUTPUT_TRANSFORM"])
def test_unknown_transform_functions_raises_a_config_error(parse_field_line, transform):
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=f"Line 3.*FIELD {transform}:silly is an invalid function",
    ):
        _ = parse_field_line(
            f"FIELD F PARAMETER f.grdecl INIT_FILES:f%d.grdecl {transform}:silly"
        )


@pytest.mark.parametrize("boolean", [True, False])
def test_forward_init_is_gotten_from_keyword(parse_field_line, boolean):
    field = parse_field_line(
        f"FIELD f PARAMETER f.roff INIT_FILES:f%d.grdecl FORWARD_INIT:{boolean}"
    )
    assert field.forward_init == boolean


@pytest.mark.parametrize("invalid", ["not_right", "uhum"])
def test_invalid_forward_init_gives_a_user_error_message(parse_field_line, invalid):
    with pytest.raises(
        ConfigValidationError,
        match=f"Invalid boolean value '{invalid}'",
    ):
        _ = parse_field_line(
            f"FIELD f PARAMETER f.roff INIT_FILES:f%d.grdecl FORWARD_INIT:{invalid}"
        )


def test_missing_init_files_gives_a_user_error_message(parse_field_line):
    with pytest.raises(
        ConfigValidationError,
        match="Line 3.*Missing required INIT_FILES for field 'foo'",
    ):
        _ = parse_field_line("FIELD foo parameter bar.roff")


def test_the_user_gets_a_warning_about_input_transform_usage(parse_field_line):
    with pytest.warns(
        ConfigWarning,
        match="Got INPUT_TRANSFORM for FIELD: f,"
        " this has no effect and can be removed",
    ):
        _ = parse_field_line(
            "FIELD f parameter out.roff INPUT_TRANSFORM:log INIT_FILES:file.init"
        )


@pytest.mark.parametrize(
    "invalid_argument", ["doesnthavecolon", "colonafter:", ":colonbefore"]
)
def test_invalid_argument_gives_a_user_error_message(
    parse_field_line, invalid_argument
):
    with pytest.raises(
        ConfigValidationError,
        match=f"Line 3.*Invalid argument '{invalid_argument}'",
    ):
        _ = parse_field_line(
            f"FIELD f parameter out.roff INIT_FILES:file.init {invalid_argument}"
        )
