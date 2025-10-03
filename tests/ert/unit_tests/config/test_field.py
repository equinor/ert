import os
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import resfo
import xtgeo

from ert.config import ConfigValidationError, ConfigWarning, Field
from ert.config.field import TRANSFORM_FUNCTIONS
from ert.config.parameter_config import InvalidParameterFile
from ert.config.parsing import init_user_config_schema, parse_contents
from ert.field_utils import FieldFileFormat, Shape, read_field
from ert.sample_prior import sample_prior


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
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
    sample_prior(prior_ensemble, active_realizations, 123)
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


def create_dummy_field(nx, ny, nz, mask):
    np.save(Path("grid_mask.npy"), mask)

    return Field(
        name="some_name",
        forward_init=True,
        update=True,
        nx=nx,
        ny=ny,
        nz=nz,
        file_format=FieldFileFormat.ROFF,
        output_transformation=None,
        input_transformation=None,
        truncation_min=None,
        truncation_max=None,
        forward_init_file="no_nees",
        output_file="no_nees",
        grid_file="no_nees",
        mask_file=Path("grid_mask.npy"),
    )


def _n_lattice_edges(nx, ny, nz):
    return ((nx - 1) * ny * nz) + (nx * (ny - 1) * nz) + (nx * ny * (nz - 1))


def test_field_grid_mask_correspondence_all_false(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    nx = 3
    ny = 3
    nz = 3
    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[:, :, :] = False

    field_config = create_dummy_field(nx=nx, ny=ny, nz=nz, mask=mask)
    graph = field_config.load_parameter_graph()

    # Expect nx*ny*nz lattice graph
    assert graph.number_of_nodes() == nx * ny * nz
    assert graph.number_of_edges() == _n_lattice_edges(nx, ny, nz)


def test_field_grid_mask_correspondence_all_true(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mask = np.ones((3, 3, 3), dtype=bool)
    field_config = create_dummy_field(nx=3, ny=3, nz=3, mask=mask)
    assert nx.node_link_data(field_config.load_parameter_graph())["links"] == []


def test_field_grid_mask_correspondence_one_false(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mask = np.ones((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = False
    field_config = create_dummy_field(nx=3, ny=3, nz=3, mask=mask)
    assert nx.node_link_data(field_config.load_parameter_graph())["links"] == []


def test_field_grid_mask_correspondence_one_true(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    nx = 3
    ny = 3
    nz = 3

    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[1, 1, 1] = True
    field_config = create_dummy_field(nx=nx, ny=ny, nz=nz, mask=mask)
    graph = field_config.load_parameter_graph()

    # Expect lattice but minus 1 node and 6 links
    assert graph.number_of_nodes() == nx * ny * nz - 1
    assert graph.number_of_edges() == _n_lattice_edges(nx, ny, nz) - 6


def test_field_grid_mask_correspondence_slice_x_true(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, :, :] = True
    field_config = create_dummy_field(nx=3, ny=3, nz=3, mask=mask)
    graph = field_config.load_parameter_graph()
    assert nx.number_connected_components(graph) == 2
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        assert subgraph.number_of_nodes() == 3 * 3
        assert subgraph.number_of_edges() == _n_lattice_edges(3, 3, 1)


def test_field_grid_mask_correspondence_slice_y_true(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[:, 1, :] = True
    field_config = create_dummy_field(nx=3, ny=3, nz=3, mask=mask)
    graph = field_config.load_parameter_graph()
    assert nx.number_connected_components(graph) == 2
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        assert subgraph.number_of_nodes() == 3 * 3
        assert subgraph.number_of_edges() == _n_lattice_edges(3, 3, 1)


def test_field_grid_mask_correspondence_slice_z_true(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[:, :, 1] = True
    field_config = create_dummy_field(nx=3, ny=3, nz=3, mask=mask)
    graph = field_config.load_parameter_graph()
    assert nx.number_connected_components(graph) == 2
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        assert subgraph.number_of_nodes() == 3 * 3
        assert subgraph.number_of_edges() == _n_lattice_edges(3, 3, 1)


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


def test_that_read_field_raises_grid_field_mismatch_error_given_different_sized_field_and_grid(  # noqa
    tmpdir, monkeypatch, parse_field_line
):
    class MockFieldFile:
        def read_keyword(self):
            return "COND"

        def read_array(self):
            return np.ndarray(shape=100, buffer=np.ones(100), dtype=float)

    def mock_lazy_read(*args):
        return [MockFieldFile()]

    monkeypatch.setattr(resfo, "lazy_read", mock_lazy_read)
    field_file_name = "foo.bgrdecl"

    field_name = "COND"
    mask = np.ndarray([])
    shape = Shape(nx=5, ny=5, nz=5)

    with tmpdir.as_cwd():
        # The file context manager needs a file to read, even if the content is mocked
        Path(field_file_name).touch()
        expected_error_message = (
            rf"The FIELD '{field_name}' from file {field_file_name} is of size \(100\) "
            r"which does not match the size of the GRID \(125\) - "
            r"derived from dimensions: \(5, 5, 5\)."
        )
        with pytest.raises(InvalidParameterFile, match=expected_error_message):
            read_field(field_file_name, field_name, mask=mask, shape=shape)
