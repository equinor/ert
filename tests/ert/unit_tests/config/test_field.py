import os
from pathlib import Path

import numpy as np
import pytest
import resfo
import xtgeo

from ert.config import ConfigValidationError, ConfigWarning, Field
from ert.config.ert_config import USER_CONFIG_SCHEMA
from ert.config.field import TRANSFORM_FUNCTIONS
from ert.config.parameter_config import InvalidParameterFile
from ert.config.parsing import parse_contents
from ert.field_utils import (
    ErtboxParameters,
    FieldFileFormat,
    ScalingFunctions,
    Shape,
    calc_rho_for_2d_grid_layer,
    calculate_ertbox_parameters,
    localization_scaling_function,
    read_field,
    transform_local_ellipse_angle_to_local_coords,
    transform_positions_to_local_field_coordinates,
)
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

    ertbox_params = ErtboxParameters(
        nx,
        ny,
        nz,
        xlength=1,
        ylength=1,
        xinc=1,
        yinc=1,
        rotation_angle=45,
        origin=(0, 0),
    )

    return Field(
        name="some_name",
        forward_init=True,
        update=True,
        ertbox_params=ertbox_params,
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


@pytest.fixture
def grid_shape():
    return Shape(2, 3, 4)


@pytest.fixture
def ertbox_params():
    return ErtboxParameters(2, 3, 4, 1, 1, 1, 1, 1, (0, 0))


@pytest.fixture
def egrid_file(tmp_path, grid_shape):
    grid = xtgeo.create_box_grid(
        dimension=(grid_shape.nx, grid_shape.ny, grid_shape.nz)
    )
    grid.to_file(tmp_path / "MY_EGRID.EGRID", "egrid")
    return str(tmp_path / "MY_EGRID.EGRID")


@pytest.fixture
def parse_field_line(ertbox_params, egrid_file):
    def make_field(field_line):
        return Field.from_config_list(
            egrid_file,
            parse_contents(
                f"NUM_REALIZATIONS 1\nGRID {egrid_file}\n" + field_line,
                USER_CONFIG_SCHEMA,
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
            read_field(field_file_name, field_name, shape=shape)


@pytest.mark.parametrize(
    "origin, increment, rotation, flip",
    [
        ((1000.0, 2000.0, 1000.0), (100.0, 150.0, 10.0), 90, 1),
        ((1000.0, 2000.0, 1000.0), (100.0, 150.0, 10.0), 45, -1),
        ((1000.0, 2000.0, 1000.0), (100.0, 150.0, 10.0), 180, 1),
        ((1000.0, 2000.0, 1000.0), (100.0, 150.0, 10.0), -30, 1),
        ((0, 0, -10.0), (1, 1, 1), 0, 1),
    ],
)
def test_calculate_ertbox_parameters_synthetic_grid(origin, increment, rotation, flip):
    """Test calculate_ertbox_parameters with a synthetic box grid."""

    # Create a synthetic box grid with rotation
    grid = xtgeo.create_box_grid(
        dimension=(5, 4, 3),  # nx, ny, nz
        origin=origin,  # x0, y0, z0
        oricenter=False,  # origin at corner, not center
        increment=increment,  # dx, dy, dz
        rotation=rotation,  # rotation in degrees
        flip=flip,  # -1 for right-handed, 1 for left-handed
    )
    params = calculate_ertbox_parameters(grid)

    # Test calculated increments (should match input within tolerance)
    tolerance = 1e-10
    expected_xinc = increment[0]
    expected_yinc = increment[1]
    assert abs(params.xinc - expected_xinc) < tolerance, (
        f"Expected xinc={expected_xinc}, got {params.xinc}"
    )
    assert abs(params.yinc - expected_yinc) < tolerance, (
        f"Expected yinc={expected_yinc}, got {params.yinc}"
    )

    # Test rotation angle (should match input within tolerance)
    expected_angle = rotation
    angle_tolerance = 1e-6  # degrees
    assert abs(params.rotation_angle - expected_angle) < angle_tolerance, (
        f"Expected rotation={expected_angle}°, got {params.rotation_angle:.6f}°"
    )

    # Test that xlength and ylength are consistent with grid dimensions and increments
    expected_xlength = params.nx * expected_xinc
    expected_ylength = params.ny * expected_yinc
    assert abs(params.xlength - expected_xlength) < tolerance, (
        f"Expected xlength={expected_xlength}, got {params.xlength}"
    )
    assert abs(params.ylength - expected_ylength) < tolerance, (
        f"Expected ylength={expected_ylength}, got {params.ylength}"
    )

    # Test that xinc and yinc are positive
    assert params.xinc > 0, f"xinc should be positive, got {params.xinc}"
    assert params.yinc > 0, f"yinc should be positive, got {params.yinc}"

    # Test that rotation angle is in reasonable range
    assert -180 <= params.rotation_angle <= 180, (
        f"Rotation angle should be between -180° and 180°, got {params.rotation_angle}"
    )

    # Test grid dimensions
    assert params.nx == 5
    assert params.ny == 4
    assert params.nz == 3


@pytest.mark.parametrize(
    "origin, rotation, utmx, utmy, expected_x, expected_y",
    [
        (
            (1000.0, 2000.0),
            0.0,
            [1100.0, 1300.0, 1500.0],
            [2000, 2100, 2500],
            [100.0, 300.0, 500.0],
            [0.0, 100.0, 500.0],
        ),
        (
            (1000.0, 2000.0),
            90.0,
            [1400.0, 1000],
            [2000.0, 2500.0],
            [0.0, 500.0],
            [-400.0, 0.0],
        ),
        (
            (1000.0, 1400.0),
            180,
            [500.0, 750.0],
            [1400.0, 1800.0],
            [500.0, 250.0],
            [0.0, -400.0],
        ),
        (
            (1000.0, 1400.0),
            30,
            [1100.0, 900.0],
            [2100.0, 1800.0],
            [436.60254037844, 113.39745962],
            [556.217782649, 396.41016151378],
        ),
    ],
)
def test_transform_positions_to_local_field_coordinates(
    origin, rotation, utmx, utmy, expected_x, expected_y
):
    """Test transformation of position from global to local coordinates."""
    tolerance = 1e-8
    xpos = np.array(utmx)
    ypos = np.array(utmy)
    x_transf, y_transf = transform_positions_to_local_field_coordinates(
        origin, rotation, xpos, ypos
    )
    print(f"{x_transf=}")
    print(f"{y_transf=}")
    reference_x = np.array(expected_x)
    reference_y = np.array(expected_y)
    for i in range(len(x_transf)):
        assert abs(x_transf[i] - reference_x[i]) < tolerance, (
            f"Expected x_transf[i]={reference_x[i]}, got {x_transf[i]}"
        )
        assert abs(y_transf[i] - reference_y[i]) < tolerance, (
            f"Expected y_transf[i]={reference_y[i]}, got {y_transf[i]}"
        )


@pytest.mark.parametrize(
    "coordsys_rotation, input_angle, expected_angle",
    [
        (0.0, 0.0, 0.0),
        (45.0, 0.0, -45.0),
        (75.0, 100.0, 25.0),
    ],
)
def test_transform_localization_ellipse_angle_to_local_coordinates(
    coordsys_rotation, input_angle, expected_angle
):
    tolerance = 1e-8
    output_angle = transform_local_ellipse_angle_to_local_coords(
        coordsys_rotation, input_angle
    )
    assert abs(output_angle - expected_angle) < tolerance


@pytest.mark.parametrize(
    "nvalues, name, expected_values",
    [
        (
            10,
            "gaspari_cohn",
            [
                1.00000000e00,
                8.87358513e-01,
                6.27163457e-01,
                3.44939558e-01,
                1.38443214e-01,
                3.52032546e-02,
                3.46364883e-03,
                2.92709888e-06,
                0.00000000e00,
                0.00000000e00,
            ],
        ),
        (
            10,
            "gaussian",
            [
                1.00000000e00,
                7.93357387e-01,
                3.96164430e-01,
                1.24514471e-01,
                2.46321272e-02,
                3.06705630e-03,
                2.40369476e-04,
                1.18569947e-05,
                3.68135480e-07,
                7.19413303e-09,
            ],
        ),
        (
            10,
            "exponential",
            [
                1.00000000e00,
                4.34598209e-01,
                1.88875603e-01,
                8.20849986e-02,
                3.56739933e-02,
                1.55038536e-02,
                6.73794700e-03,
                2.92829969e-03,
                1.27263380e-03,
                5.53084370e-04,
            ],
        ),
    ],
)
def test_localization_scaling_function(nvalues: int, name: str, expected_values: list):
    tolerance = 1e-8
    distances = np.linspace(0, 2.5, num=nvalues, endpoint=True, dtype=np.float64)
    values = localization_scaling_function(
        distances, scaling_func=ScalingFunctions(name)
    )
    reference_values = np.array(expected_values, dtype=np.float64)
    are_equal = np.allclose(values, reference_values, rtol=0.01, atol=tolerance)
    assert are_equal


@pytest.mark.parametrize(
    "xpos, ypos, main_range, perp_range, anisotropy_angle, scaling_func, expected",
    [
        (
            [50.0, 150.0, 250.0],  # xpos
            [50.0, 150.0, 250.0],  # ypos
            [150.0, 150.0, 150.0],  # main_range
            [100.0, 100.0, 100.0],  # perp_range
            [0.0, 45.0, 90.0],  # angle
            "gaussian",
            [
                [
                    [1.42516408e-21, 2.62309377e-12, 2.96639500e-08],
                    [1.87952882e-12, 7.03873864e-07, 1.61959679e-06],
                    [6.14421235e-06, 2.47875218e-03, 6.14421235e-06],
                    [4.97870684e-02, 1.14558844e-01, 1.61959679e-06],
                    [1.00000000e00, 6.94834512e-02, 2.96639500e-08],
                ],
                [
                    [3.75669174e-22, 3.39826782e-09, 2.40369476e-04],
                    [4.95438417e-13, 1.72232256e-04, 1.31237287e-02],
                    [1.61959679e-06, 1.14558844e-01, 4.97870684e-02],
                    [1.31237287e-02, 1.00000000e00, 1.31237287e-02],
                    [2.63597138e-01, 1.14558844e-01, 2.40369476e-04],
                ],
                [
                    [6.88062092e-24, 5.77774852e-08, 4.82794999e-03],
                    [9.07427114e-15, 5.53084370e-04, 2.63597138e-01],
                    [2.96639500e-08, 6.94834512e-02, 1.00000000e00],
                    [2.40369476e-04, 1.14558844e-01, 2.63597138e-01],
                    [4.82794999e-03, 2.47875218e-03, 4.82794999e-03],
                ],
                [
                    [8.75651076e-27, 1.28918995e-08, 2.40369476e-04],
                    [1.15482242e-17, 2.33091011e-05, 1.31237287e-02],
                    [3.77513454e-11, 5.53084370e-04, 4.97870684e-02],
                    [3.05902321e-07, 1.72232256e-04, 1.31237287e-02],
                    [6.14421235e-06, 7.03873864e-07, 2.40369476e-04],
                ],
                [
                    [7.74311878e-31, 3.77513454e-11, 2.96639500e-08],
                    [1.02117469e-21, 1.28918995e-08, 1.61959679e-06],
                    [3.33823780e-15, 5.77774852e-08, 6.14421235e-06],
                    [2.70500210e-11, 3.39826782e-09, 1.61959679e-06],
                    [5.43314196e-10, 2.62309377e-12, 2.96639500e-08],
                ],
            ],
        ),
        (
            [150.0, 150.0, 250.0],  # xpos
            [350.0, 150.0, 250.0],  # ypos
            [150.0, 450.0, 450.0],  # main_range
            [100.0, 200.0, 300.0],  # perp_range
            [0.0, 35.0, 135.0],  # angle
            "gaspari_cohn",
            [
                [
                    [9.42127705e-02, 1.31635111e-02, 5.50270948e-01],
                    [5.10288066e-01, 1.39972670e-01, 6.56951586e-01],
                    [9.42127705e-02, 4.77285176e-01, 6.15604839e-01],
                    [0.00000000e00, 8.37715936e-01, 4.50850465e-01],
                    [0.00000000e00, 8.43419782e-01, 2.51129163e-01],
                ],
                [
                    [2.08333333e-01, 6.24711885e-02, 6.56951586e-01],
                    [1.00000000e00, 3.22638741e-01, 8.58901220e-01],
                    [2.08333333e-01, 7.55961390e-01, 8.83226538e-01],
                    [-2.77555756e-16, 1.00000000e00, 7.13974029e-01],
                    [0.00000000e00, 7.55961390e-01, 4.50850465e-01],
                ],
                [
                    [9.42127705e-02, 1.45718716e-01, 6.15604839e-01],
                    [5.10288066e-01, 4.87371035e-01, 8.83226538e-01],
                    [9.42127705e-02, 8.43419782e-01, 1.00000000e00],
                    [0.00000000e00, 8.37715936e-01, 8.83226538e-01],
                    [0.00000000e00, 4.77285176e-01, 6.15604839e-01],
                ],
                [
                    [3.46364883e-03, 2.13194348e-01, 4.50850465e-01],
                    [4.86968450e-02, 5.11061537e-01, 7.13974029e-01],
                    [3.46364883e-03, 6.66165595e-01, 8.83226538e-01],
                    [0.00000000e00, 4.97072413e-01, 8.58901220e-01],
                    [0.00000000e00, 2.00473342e-01, 6.56951586e-01],
                ],
                [
                    [0.00000000e00, 2.09121884e-01, 2.51129163e-01],
                    [-2.77555756e-16, 3.74226141e-01, 4.50850465e-01],
                    [0.00000000e00, 3.66249824e-01, 6.15604839e-01],
                    [0.00000000e00, 1.95100370e-01, 6.56951586e-01],
                    [0.00000000e00, 4.76659917e-02, 5.50270948e-01],
                ],
            ],
        ),
    ],
)
def test_calc_rho_for_2d_grid_layer(
    xpos: list[float],
    ypos: list[float],
    main_range: list[float],
    perp_range: list[float],
    anisotropy_angle: list[float],
    scaling_func: str,
    expected: list[list[list[float]]],
):
    nx = 5
    ny = 5
    xinc = 100.0
    yinc = 100.0
    tolerance = 1e-8

    xposition = np.array(xpos)
    yposition = np.array(ypos)
    mainrange = np.array(main_range)
    perprange = np.array(perp_range)
    angles = np.array(anisotropy_angle)
    reference_values = np.array(expected)
    rho_for_one_grid_layer = calc_rho_for_2d_grid_layer(
        nx,
        ny,
        xinc,
        yinc,
        xposition,
        yposition,
        mainrange,
        perprange,
        angles,
        scaling_function=ScalingFunctions(scaling_func),
    )
    are_equal = np.allclose(
        rho_for_one_grid_layer, reference_values, rtol=0.01, atol=tolerance
    )
    assert are_equal
