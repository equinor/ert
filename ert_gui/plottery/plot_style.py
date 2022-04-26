class PlotStyle:
    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        name,
        color="#000000",
        alpha=1.0,
        line_style="-",
        marker="",
        width=1.0,
        size=7.5,
        enabled=True,
    ):
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

    def copyStyleFrom(self, other, copy_enabled_state=False):
        self.color = other.color
        self.alpha = other.alpha
        self.line_style = other._line_style
        self.marker = other._marker
        self.width = other.width
        self.size = other.size
        self._is_copy = True

        if copy_enabled_state:
            self.setEnabled(other.isEnabled())

    def isEnabled(self):
        return self._enabled

    def setEnabled(self, enabled):
        self._enabled = enabled

    def isVisible(self):
        return self.line_style != "" or self.marker != ""

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = max(min(alpha, 1.0), 0.0)

    @property
    def marker(self):
        return self._marker if self._marker is not None else ""

    @marker.setter
    def marker(self, marker):
        self._marker = marker

    @property
    def line_style(self):
        return self._line_style if self._line_style is not None else ""

    @line_style.setter
    def line_style(self, line_style):
        self._line_style = line_style

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = max(width, 0.0)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = max(size, 0.0)

    def __str__(self):
        return (
            f"{self.name} c:{self.color} a:{self.alpha} "
            f"ls:{self.line_style} m:{self.marker} w:{self.width} "
            f"s:{self.size} enabled:{self.isEnabled()} copy:{self._is_copy}"
        )

    def __eq__(self, other):
        equalness = self.alpha == other.alpha
        equalness = equalness and self.marker == other.marker
        equalness = equalness and self.line_style == other.line_style
        equalness = equalness and self.width == other.width
        equalness = equalness and self.color == other.color
        equalness = equalness and self.size == other.size
        equalness = equalness and self.isEnabled() == other.isEnabled()

        return equalness
