"""Centralized layout metrics (spacing tokens + a layout helper) for the GUI.

Layout geometry — the margins a :class:`~PyQt6.QtWidgets.QLayout` reserves and
the spacing it distributes between child widgets — cannot be expressed in Qt
Style Sheets: QSS styles a widget's own box model (background, border, padding,
font, colour) but has no selector for a layout manager and no property for
inter-child spacing or layout contents-margins. Those values therefore have to
live in Python.

This module gives that Python-side geometry a single home and a shared
vocabulary, mirroring how :mod:`ert.gui.theming.eds` centralizes colour tokens.
Pages import the ``SPACE_*`` tokens and :func:`apply_layout` instead of hardcoding
raw pixel numbers, so the spacing scale can be tuned in one place.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QLayout

# Spacing scale aligned to the Equinor Design System spacing steps. Dimensions do
# not vary by colour scheme, so these are plain constants rather than
# scheme-keyed tokens like the EDS colours.
SPACE_NONE = 0
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 16
SPACE_LG = 24
SPACE_XL = 32

# Left, top, right, bottom contents margins for a layout.
Margins = tuple[int, int, int, int]


def apply_layout(layout: QLayout, *, margins: int | Margins, spacing: int) -> None:
    """Set a layout's contents margins and inter-child spacing in one call.

    Args:
        layout: The layout to configure.
        margins: Either a single value applied uniformly to all four edges, or a
            ``(left, top, right, bottom)`` tuple.
        spacing: The gap the layout distributes between adjacent child widgets.
    """
    if isinstance(margins, int):
        margins = (margins, margins, margins, margins)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
