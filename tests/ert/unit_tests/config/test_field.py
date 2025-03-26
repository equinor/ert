import os
from pathlib import Path

import networkx as nx
import numpy as np
import orjson
import pytest
import xtgeo

from ert.config import ConfigValidationError, ConfigWarning, Field
from ert.config.field import TRANSFORM_FUNCTIONS
from ert.config.parsing import init_user_config_schema, parse_contents
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
def snake_oil_field_config(snake_oil_field_example, storage):
    ensemble_config = snake_oil_field_example.ensemble_config
    field_config = ensemble_config["PERMX"]
    experiment = storage.create_experiment(
        parameters=ensemble_config.parameter_configuration
    )
    ensemble = storage.create_ensemble(experiment, name="prior", ensemble_size=5)

    # Downscale for smaller snapshots
    field_config.nx = 2
    field_config.ny = 2
    field_config.nz = 2

    field_config.save_experiment_data(experiment._path)
    os.remove(field_config.mask_file)
    np.save(field_config.mask_file, np.full((2, 2, 2), False, dtype=bool))

    return field_config, ensemble


def create_graph_snapshot(field_config, ensemble):
    graph = field_config.load_parameter_graph(
        ensemble, "PERMX", np.array(range(ensemble.ensemble_size))
    )
    mask = field_config.mask

    return (
        orjson.dumps(
            {**nx.node_link_data(graph), "mask": mask.tolist()},
            option=orjson.OPT_INDENT_2,
        )
        .decode("utf-8")
        .strip()
        + "\n"
    )


def test_field_grid_mask_correspondence_all_false(snake_oil_field_config, snapshot):
    field_config, ensemble = snake_oil_field_config

    field_config.mask[:, :, :] = False
    snapshot.assert_match(create_graph_snapshot(field_config, ensemble), "graph.json")


def test_field_grid_mask_correspondence_all_true(snake_oil_field_config, snapshot):
    field_config, ensemble = snake_oil_field_config

    field_config.mask[:, :, :] = True
    snapshot.assert_match(create_graph_snapshot(field_config, ensemble), "graph.json")


def test_field_grid_mask_correspondence_one_false(snake_oil_field_config, snapshot):
    field_config, ensemble = snake_oil_field_config

    field_config.mask[:, :, :] = True
    field_config.mask[1, 1, 1] = False

    snapshot.assert_match(create_graph_snapshot(field_config, ensemble), "graph.json")


def test_field_grid_mask_correspondence_one_true(snake_oil_field_config, snapshot):
    field_config, ensemble = snake_oil_field_config

    field_config.mask[1, 1, 1] = True
    snapshot.assert_match(create_graph_snapshot(field_config, ensemble), "graph.json")


def test_field_grid_mask_correspondence_slice_x_true(snake_oil_field_config, snapshot):
    field_config, ensemble = snake_oil_field_config

    field_config.mask[0, :, :] = True
    snapshot.assert_match(create_graph_snapshot(field_config, ensemble), "graph.json")


def test_field_grid_mask_correspondence_slice_y_true(snake_oil_field_config, snapshot):
    field_config, ensemble = snake_oil_field_config

    field_config.mask[:, 0, :] = True
    snapshot.assert_match(create_graph_snapshot(field_config, ensemble), "graph.json")


def test_field_grid_mask_correspondence_slice_z_true(snake_oil_field_config, snapshot):
    field_config, ensemble = snake_oil_field_config

    field_config.mask[:, :, 0] = True
    snapshot.assert_match(create_graph_snapshot(field_config, ensemble), "graph.json")


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
def parse_field_line(grid_shape, egrid_file):
    def make_field(field_line):
        return Field.from_config_list(
            egrid_file,
            grid_shape,
            parse_contents(
                f"NUM_REALIZATIONS 1\nGRID {egrid_file}\n" + field_line,
                init_user_config_schema(),
                "test.ert",
            )["FIELD"][0],
        )

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
        match=r"Line 3.*Missing required INIT_FILES for field 'foo'",
    ):
        _ = parse_field_line("FIELD foo parameter bar.roff")


def test_the_user_gets_a_warning_about_input_transform_usage(parse_field_line):
    with pytest.warns(
        ConfigWarning,
        match="Got INPUT_TRANSFORM for FIELD: f, this has no effect and can be removed",
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
