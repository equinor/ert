from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ShapeConfig(BaseModel):
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


Shape = Annotated[CircleShapeConfig, Field(discriminator="type")]


class ShapeRegistry(BaseModel):
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

        if not isinstance(shape, CircleShapeConfig):
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
