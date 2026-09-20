from __future__ import annotations

from importlib.resources import files

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from .detect_mode import is_dark_mode

# Render SVGs at this resolution so Qt always has plenty of pixels to scale
# down from, regardless of the button's actual icon size.
_RENDER_SIZE = 256


def load_icon(name: str) -> QIcon:
    """Load an SVG icon tinted for the current colour theme.

    ``QPixmap.loadFromData`` rasterises at the SVG's natural size (24x24).\n
    Qt's QIcon does not upscale a pixmap to match
    ``setIconSize``, so the icon would appear tiny.  Using ``QSvgRenderer``
    to render at a high resolution first means Qt always has crisp pixels
    available to scale down to whatever size the widget requests.
    """
    color = "white" if is_dark_mode() else "black"
    path = files("ert.gui").joinpath(f"resources/gui/img/{name}")
    data = path.read_bytes().replace(b"currentColor", color.encode())

    renderer = QSvgRenderer(bytes(data))
    pixmap = QPixmap(_RENDER_SIZE, _RENDER_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)
