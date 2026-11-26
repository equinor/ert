from __future__ import annotations

from importlib.resources import files

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from .detect_mode import is_dark_mode

# Render SVGs at this resolution so Qt always has plenty of pixels to scale
# down from, regardless of the button's actual icon size.
_RENDER_SIZE = 256


def load_icon(name: str, color: str | None = None) -> QIcon:
    """Load an SVG icon tinted for the current colour theme.

    ``QPixmap.loadFromData`` rasterises at the SVG's natural size (24x24).\n
    Qt's QIcon does not upscale a pixmap to match
    ``setIconSize``, so the icon would appear tiny.  Using ``QSvgRenderer``
    to render at a high resolution first means Qt always has crisp pixels
    available to scale down to whatever size the widget requests.
    """
    color = color or ("white" if is_dark_mode() else "#3d3d3d")
    return QIcon(_render_pixmap(name, color))


def _render_pixmap(name: str, color: str, trailing_gap: float = 0.0) -> QPixmap:
    path = files("ert.gui").joinpath(f"resources/eds/icons/{name}")
    if not path.is_file():
        path = files("ert.gui").joinpath(f"resources/gui/img/{name}")
    data = path.read_bytes().replace(b"currentColor", color.encode())

    renderer = QSvgRenderer(bytes(data))
    pixmap = QPixmap(round(_RENDER_SIZE * (1 + trailing_gap)), _RENDER_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, _RENDER_SIZE, _RENDER_SIZE))
    painter.end()
    return pixmap


def load_stateful_icon(
    name: str,
    resting: str,
    active: str,
    disabled: str,
    trailing_gap: float = 0.0,
) -> QIcon:
    """Load an SVG icon tinted per state (resting, active and disabled).

    The On state is used by checked (active) buttons. ``trailing_gap`` adds
    transparent space to the right of the glyph, as a fraction of its width,
    to widen the gap to a label drawn beside the icon; set the widget's icon
    size to the matching aspect ratio.
    """
    icon = QIcon()
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
        icon.addPixmap(
            _render_pixmap(name, resting, trailing_gap), mode, QIcon.State.Off
        )
        icon.addPixmap(_render_pixmap(name, active, trailing_gap), mode, QIcon.State.On)
    for state in (QIcon.State.Off, QIcon.State.On):
        icon.addPixmap(
            _render_pixmap(name, disabled, trailing_gap), QIcon.Mode.Disabled, state
        )
    return icon
