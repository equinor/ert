from textwrap import dedent

import pytest

from ert.config._shapes import CircleShapeConfig, PolygonShapeConfig, ShapeRegistry


def test_that_shape_registry_reuses_identical_circle_shapes():
    shape_registry = ShapeRegistry()
    shape_id_1 = shape_registry.register(
        CircleShapeConfig(east=10.0, north=20.0, radius=2500.0)
    )
    shape_id_2 = shape_registry.register(
        CircleShapeConfig(east=10.0, north=20.0, radius=2500.0)
    )
    assert shape_id_1 == shape_id_2


def test_that_shape_registry_assigns_new_id_for_different_shapes():
    shape_registry = ShapeRegistry()
    shape_id_1 = shape_registry.register(
        CircleShapeConfig(east=10.0, north=20.0, radius=2500.0)
    )
    shape_id_2 = shape_registry.register(
        CircleShapeConfig(east=10.0, north=20.0, radius=3000.0)
    )
    assert shape_id_1 != shape_id_2


def test_that_polygon_shape_is_read_from_file(mocked_files):
    mocked_files["polygon.pol"] = dedent(
        """
        0.500000 4.000000 0.000000
        1.000000 7.000000 0.000000
        2.000000 9.000000 0.000000
        3.000000 6.000000 0.000000
        1.000000 5.000000 0.000000
        0.500000 4.000000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    shape = PolygonShapeConfig.from_file("polygon.pol")
    expected = [
        (0.5, 4.0),
        (1.0, 7.0),
        (2.0, 9.0),
        (3.0, 6.0),
        (1.0, 5.0),
        (0.5, 4.0),
    ]
    assert shape.vertices == expected


def test_that_polygon_shape_is_normalized(mocked_files):
    mocked_files["polygon.pol"] = dedent(
        """
        2.000000 5.000000 0.000000
        4.000000 6.000000 0.000000
        3.000000 9.000000 0.000000
        1.000000 6.000000 0.000000
        2.000000 5.000000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    shape = PolygonShapeConfig.from_file("polygon.pol")
    expected = [
        (1.0, 6.0),
        (3.0, 9.0),
        (4.0, 6.0),
        (2.0, 5.0),
        (1.0, 6.0),
    ]
    assert shape.vertices == expected


def test_that_polygon_shape_is_simplified_within_tolerance(mocked_files):
    mocked_files["polygon.pol"] = dedent(
        """
        0.000000 0.000000 0.000000
        0.000000 1.500000 0.000000
        0.000000 2.000000 0.000000
        1.000000 2.200000 0.000000
        2.000000 2.000000 0.000000
        2.050000 1.000000 0.000000
        2.000000 0.000000 0.000000
        1.000000 0.000000 0.000000
        0.000000 0.000000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    shape = PolygonShapeConfig.from_file("polygon.pol")
    expected = [
        (0.0, 0.0),
        (0.0, 2.0),
        (1.0, 2.2),
        (2.0, 2.0),
        (2.0, 0.0),
        (0.0, 0.0),
    ]
    assert shape.vertices == expected


def test_that_polygon_shape_raises_when_multiple_polygons_found(mocked_files):
    mocked_files["polygon.pol"] = dedent(
        """
        0.000000 0.000000 0.000000
        0.000000 1.000000 0.000000
        1.000000 1.000000 0.000000
        1.000000 0.000000 0.000000
        0.000000 0.000000 0.000000
        999.000000 999.000000 999.000000

        2.000000 2.000000 0.000000
        2.000000 3.000000 0.000000
        3.000000 3.000000 0.000000
        3.000000 2.000000 0.000000
        2.000000 2.000000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    with pytest.raises(ValueError, match="Multiple polygons found in the file"):
        PolygonShapeConfig.from_file("polygon.pol")


def test_that_polygon_shape_raises_when_geometry_not_a_polygon(mocked_files):
    mocked_files["polygon.pol"] = dedent(
        """
        0.000000 0.000000 0.000000
        0.000000 1.000000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    with pytest.raises(ValueError, match="Failed to create polygon from file"):
        PolygonShapeConfig.from_file("polygon.pol")


def test_that_polygons_are_equal_within_tolerance(mocked_files):
    mocked_files["polygon1.pol"] = dedent(
        """
        0.000000 0.000000 0.000000
        0.000000 1.000000 0.000000
        1.000000 1.000000 0.000000
        1.000000 0.000000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    shape1 = PolygonShapeConfig.from_file("polygon1.pol")

    mocked_files["polygon2.pol"] = dedent(
        """
        0.050000 0.050000 0.000000
        0.050000 1.050000 0.000000
        1.050000 1.050000 0.000000
        1.050000 0.050000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    shape2 = PolygonShapeConfig.from_file("polygon2.pol")

    assert shape1 == shape2


def test_that_polygons_are_not_equal_outside_tolerance(mocked_files):
    mocked_files["polygon1.pol"] = dedent(
        """
        0.000000 0.000000 0.000000
        0.000000 1.000000 0.000000
        1.000000 1.000000 0.000000
        1.000000 0.000000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    shape1 = PolygonShapeConfig.from_file("polygon1.pol")

    mocked_files["polygon2.pol"] = dedent(
        """
        0.150000 0.150000 0.000000
        0.150000 1.150000 0.000000
        1.150000 1.150000 0.000000
        1.150000 0.150000 0.000000
        999.000000 999.000000 999.000000
        """
    )
    shape2 = PolygonShapeConfig.from_file("polygon2.pol")

    assert shape1 != shape2
