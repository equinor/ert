class PlotStyle(object):
    def __init__(self, name, color="#000000", alpha=1.0, line_style="-", marker="", width=1.0):
        super(PlotStyle, self).__init__()
        self.name = name
        self.color = color
        self.alpha = alpha
        self.line_style = line_style
        self.marker = marker
        self.width = width    # todo: differentiate between line_width and marker_size?
        self.__enabled = True
        self.__is_copy = False

    def copyStyleFrom(self, other, copy_enabled_state=False):
        self.color = other.color
        self.alpha = other.alpha
        self.line_style = other.line_style
        self.marker = other.__marker
        self.width = other.width
        self.__is_copy = True

        if copy_enabled_state:
            self.setEnabled(other.isEnabled())

    def isEnabled(self):
        return self.__enabled

    def setEnabled(self, enabled):
        self.__enabled = enabled

    def isVisible(self):
        return self.line_style != "" or self.marker != ""

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def color(self):
        return self.__color

    @color.setter
    def color(self, color):
        self.__color = color

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha > 1.0:
            alpha = 1.0
        if alpha < 0.0:
            alpha = 0.0
        self.__alpha = alpha

    @property
    def marker(self):
        return self.__marker if self.__marker is not None else ""

    @marker.setter
    def marker(self, marker):
        self.__marker = marker

    @property
    def line_style(self):
        return self.__line_style if self.__line_style is not None else ""

    @line_style.setter
    def line_style(self, line_style):
        self.__line_style = line_style

    @property
    def width(self):
        return self.__width if self.__width is not None else ""

    @width.setter
    def width(self, width):
        if width < 0.0:
            width = 0.0
        self.__width = width

    def __str__(self):
        return "%s c:%s a:%f ls:%s m:%s w:%f enabled:%s copy:%s" % (self.name, self.color, self.alpha, self.line_style, self.marker, self.width, self.isEnabled(), self.__is_copy)

    def __eq__(self, other):
        equalness = self.alpha == other.alpha
        equalness = equalness and self.marker == other.marker
        equalness = equalness and self.line_style == other.line_style
        equalness = equalness and self.width == other.width
        equalness = equalness and self.color == other.color
        equalness = equalness and self.isEnabled() == other.isEnabled()

        return equalness