from __future__ import annotations

from importlib.resources import files

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap, QTransform
from PyQt6.QtSvg import QSvgRenderer

from .detect_mode import is_dark_mode

# Render SVGs at this resolution so Qt always has plenty of pixels to scale
# down from, regardless of the button's actual icon size.
_RENDER_SIZE = 256


def load_icon(name: str, rotation: float = 0.0, color: str | None = None) -> QIcon:
    """Load an SVG icon tinted for the current colour theme.

    ``QPixmap.loadFromData`` rasterises at the SVG's natural size (24x24).\n
    Qt's QIcon does not upscale a pixmap to match
    ``setIconSize``, so the icon would appear tiny.  Using ``QSvgRenderer``
    to render at a high resolution first means Qt always has crisp pixels
    available to scale down to whatever size the widget requests.

    Args:
        name: File name of the SVG under ``resources/gui/img``.
        rotation: Rotation in degrees applied to the rendered icon (positive
            is clockwise on screen, since Qt's y-axis points downward).
        color: Explicit tint applied to the SVG's ``currentColor`` (any value
            Qt accepts, e.g. ``"#141f20"``). When ``None`` the icon falls back
            to plain white/black based on the detected OS colour scheme.
    """
    if color is None:
        color = "white" if is_dark_mode() else "black"
    path = files("ert.gui").joinpath(f"resources/gui/img/{name}")
    data = path.read_bytes().replace(b"currentColor", color.encode())

    renderer = QSvgRenderer(bytes(data))
    pixmap = QPixmap(_RENDER_SIZE, _RENDER_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    if rotation:
        pixmap = pixmap.transformed(QTransform().rotate(rotation))
    return QIcon(pixmap)
