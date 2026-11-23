from __future__ import annotations

from importlib.resources import files

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
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
    path = files("ert.gui").joinpath(f"resources/gui/img/{name}")
    data = path.read_bytes().replace(b"currentColor", color.encode())

    renderer = QSvgRenderer(bytes(data))
    pixmap = QPixmap(_RENDER_SIZE, _RENDER_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


def _badge_pixmap(name: str, background: str) -> QPixmap:
    glyph_size = _RENDER_SIZE * 3 // 5
    path = files("ert.gui").joinpath(f"resources/gui/img/{name}")
    data = path.read_bytes().replace(b"currentColor", b"white")
    renderer = QSvgRenderer(bytes(data))

    pixmap = QPixmap(_RENDER_SIZE, _RENDER_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(background))
    radius = _RENDER_SIZE // 5
    painter.drawRoundedRect(0, 0, _RENDER_SIZE, _RENDER_SIZE, radius, radius)
    offset = (_RENDER_SIZE - glyph_size) // 2
    renderer.render(painter, QRectF(offset, offset, glyph_size, glyph_size))
    painter.end()
    return pixmap


def load_badge_icon(name: str) -> QIcon:
    """Load an SVG icon as a white glyph on a green badge.

    The badge is darker when the icon is in the On state (checked buttons) and
    grey when disabled.
    """
    if is_dark_mode():
        return load_icon(name)
    icon = QIcon()
    icon.addPixmap(_badge_pixmap(name, "#007079"), QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(_badge_pixmap(name, "#004f55"), QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(_badge_pixmap(name, "#bebebe"), QIcon.Mode.Disabled, QIcon.State.Off)
    return icon
