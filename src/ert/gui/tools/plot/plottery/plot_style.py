from typing import Optional


class PlotStyle:
    def __init__(
        self,
        name: str,
        color: Optional[str] = "#000000",
        alpha: float = 1.0,
        line_style: str = "-",
        marker: str = "",
        width: float = 1.0,
        size: float = 7.5,
        enabled: bool = True,
    ) -> None:
        super().__init__()

        self.name = name
        self.color = color
        self.alpha = alpha
        self.line_style = line_style
        self.marker = marker
        self.width = width
        self.size = size
        self._enabled = enabled
        self._is_copy = False

    def copyStyleFrom(
        self, other: "PlotStyle", copy_enabled_state: bool = False
    ) -> None:
        self.color = other.color
        self.alpha = other.alpha
        self.line_style = other._line_style
        self.marker = other._marker
        self.width = other.width
        self.size = other.size
        self._is_copy = True

        if copy_enabled_state:
            self.setEnabled(other.isEnabled())

    def isEnabled(self) -> bool:
        return self._enabled

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def isVisible(self) -> bool:
        return bool(self.line_style or self.marker)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = max(min(alpha, 1.0), 0.0)

    @property
    def marker(self) -> str:
        return self._marker if self._marker is not None else ""

    @marker.setter
    def marker(self, marker: str) -> None:
        self._marker = marker

    @property
    def line_style(self) -> str:
        return self._line_style if self._line_style is not None else ""

    @line_style.setter
    def line_style(self, line_style: str) -> None:
        self._line_style = line_style

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, width: float) -> None:
        self._width = max(width, 0.0)

    @property
    def size(self) -> float:
        return self._size

    @size.setter
    def size(self, size: float) -> None:
        self._size = max(size, 0.0)

    def __str__(self) -> str:
        return (
            f"{self.name} c:{self.color} a:{self.alpha} "
            f"ls:{self.line_style} m:{self.marker} w:{self.width} "
            f"s:{self.size} enabled:{self.isEnabled()} copy:{self._is_copy}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlotStyle):
            return False

        equalness = self.alpha == other.alpha
        equalness = equalness and self.marker == other.marker
        equalness = equalness and self.line_style == other.line_style
        equalness = equalness and self.width == other.width
        equalness = equalness and self.color == other.color
        equalness = equalness and self.size == other.size
        equalness = equalness and self.isEnabled() == other.isEnabled()

        return equalness
