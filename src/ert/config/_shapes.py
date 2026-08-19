from __future__ import annotations

from functools import cached_property
from typing import Annotated, ClassVar, Literal, Self

import shapely
import xtgeo
from pydantic import BaseModel, Field


class ShapeConfig(BaseModel, extra="forbid"):
    """Base class for all shape configurations models for observations."""

    shape_id: int | None = None


class CircleShapeConfig(ShapeConfig):
    """Configuration for a circular (point-based) shape.

    Attributes:
        shape_id: Unique identifier for this shape, if registered.
        east: X-coordinate of the circle center (meters).
        north: Y-coordinate of the circle center (meters).
        radius: Radius of localization in meters.
    """

    type: Literal["circle"] = "circle"
    east: float
    north: float
    radius: float

    def __eq__(self, other: object) -> bool:
        """Compare two CircleShapeConfig instances by geometry."""
        if not isinstance(other, CircleShapeConfig):
            return False
        return (
            self.east == other.east
            and self.north == other.north
            and self.radius == other.radius
        )


class PolygonShapeConfig(ShapeConfig):
    """Configuration for a polygonal shape.

    Attributes:
        vertices: List of (east, north) tuples defining the polygon vertices. Vertices
        are expected to be normalized in shapely's sense (first vertex is the lowest,
        and vertices are ordered clockwise).
    """

    type: Literal["polygon"] = "polygon"
    vertices: list[tuple[float, float]]

    TOLERANCE: ClassVar[float] = 0.1

    @classmethod
    def from_file(cls, filepath: str) -> Self:
        """Create a PolygonShapeConfig from a file containing polygon vertices.

        Args:
            filepath: Path to a file containing polygon definition. Supported formats
            are the ones supported by xtgeo.polygons_from_file
            https://xtgeo.readthedocs.io/en/latest/api-points-polygons.html#xtgeo.polygons_from_file.
            Expected to contain exactly one polygon with no holes.
        """
        xtgeo_polygon = xtgeo.polygons_from_file(filepath)

        if len(set(xtgeo_polygon.dataframe["POLY_ID"])) != 1:
            raise ValueError(
                "Multiple polygons found in the file. "
                "Behavior is defined only for one polygon."
            )

        try:
            shapely_polygon = (
                shapely.Polygon(xtgeo_polygon.get_xyz_arrays())
                .simplify(tolerance=cls.TOLERANCE)
                .normalize()
            )
        except Exception as e:
            raise ValueError(f"Failed to create polygon from file {filepath}") from e

        vertices = shapely.get_coordinates(shapely_polygon).tolist()

        return cls(vertices=vertices)

    @cached_property
    def _polygon(self) -> shapely.Polygon:
        poly = shapely.Polygon(self.vertices)
        shapely.prepare(poly)
        return poly

    def contains(self, east: float, north: float) -> bool:
        """Check if a point is inside the polygon.

        Args:
            east: UTM_X-coordinate of the point
            north: UTM_Y-coordinate of the point

        Returns:
            True if the point is inside the polygon (but not on the boundary), False
            otherwise. Behavior is like that (and without any tolerance applied) as it
            is assumed it doesn't matter much what happens to the points near the
            boundary.
        """
        return bool(shapely.contains_xy(self._polygon, east, north))

    def __eq__(self, other: object) -> bool:
        """Compare two PolygonShapeConfig instances by geometry.

        Polygons are considered equal if their vertices are equal within a tolerance.
        """
        if not isinstance(other, PolygonShapeConfig):
            return False
        return self._polygon.equals_exact(other._polygon, tolerance=self.TOLERANCE)


Shape = Annotated[CircleShapeConfig | PolygonShapeConfig, Field(discriminator="type")]


class ShapeRegistry(BaseModel, extra="forbid"):
    """Registry for reusable shape configurations.

    Resolves identical geometries and assigns unique shape IDs.

    Attributes:
        shapes: Mapping from shape_id to ShapeConfig (serialized).
    """

    shapes: dict[int, Shape] = Field(default_factory=dict)

    def register(self, shape: ShapeConfig) -> int:
        """Register or find an existing shape.

        Checks if a shape with identical geometry already exists.
        If so, returns its shape_id. Otherwise, stores a copy with a new ID and
        returns that ID.

        Args:
            shape: Shape configuration to register. The input shape may have
                ``shape_id=None`` and will be copied with the assigned ID.

        Returns:
            Unique shape_id for this geometry.
        """

        if not isinstance(shape, (CircleShapeConfig, PolygonShapeConfig)):
            msg = f"Unsupported shape config type: {type(shape).__name__}"
            raise TypeError(msg)

        # Check for existing shape with same geometry
        for existing_id, existing_shape in self.shapes.items():
            if shape == existing_shape:
                return existing_id

        new_id = max(self.shapes.keys(), default=-1) + 1
        self.shapes[new_id] = shape.model_copy(update={"shape_id": new_id})
        return new_id

    def get(self, shape_id: int) -> Shape | None:
        """Retrieve a shape by its ID.

        Args:
            shape_id: The shape identifier.

        Returns:
            The ShapeConfig if found, None otherwise.
        """
        return self.shapes.get(shape_id)
