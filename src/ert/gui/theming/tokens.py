from __future__ import annotations

from dataclasses import dataclass

from .eds import semantic
from .theme import ColorScheme

# Mapping from a stylesheet template variable (used in ``theme.qss.in``) to the
# Equinor Design System *semantic* token that fulfils that role. Colours are
# never written as raw hex here: each purpose points at an EDS token by name, so
# the shade is chosen by the design system and resolved per colour scheme. Edit
# these mappings to re-purpose a variable; run ``ert.gui.theming.eds.sync`` to
# refresh the underlying EDS values.
_SEMANTIC_MAP: dict[str, str] = {
    # Window / global surfaces
    "background-primary": "bg-neutral-canvas",
    "text-primary": "text-neutral-strong",
    # Sidebar container
    "sidebar-bg": "bg-neutral-surface",
    # Sidebar navigation buttons, one mapping per interaction state
    "nav-bg": "bg-neutral-surface",
    "nav-text": "text-neutral-subtle",
    "nav-focus-bg": "bg-neutral-surface",
    "nav-focus-text": "text-neutral-strong",
    "nav-hover-bg": "bg-neutral-fill-muted-hover",
    "nav-hover-text": "text-neutral-strong",
    "nav-active-bg": "bg-accent-fill-muted-default",
    "nav-active-text": "text-accent-subtle",
    "nav-active-border": "border-accent-strong",
    "nav-pressed-bg": "bg-neutral-fill-muted-active",
    "nav-pressed-text": "text-neutral-strong",
    "nav-disabled-bg": "bg-disabled",
    "nav-disabled-text": "text-disabled",
}


@dataclass(frozen=True)
class Typography:
    """Font sizes (px) for the sidebar's text roles.

    Values come from the Equinor Design System typography tokens
    (eds-tokens, comfortable/default density). ``title`` maps to
    ``typography.header.lg``, ``nav_label`` to ``typography.ui.lg`` (the size
    EDS uses for the sidebar destination item) and ``menu_item`` to
    ``typography.ui.md``.
    """

    title: int
    nav_label: int
    menu_item: int


# Single source of truth for the sidebar font sizes; shared across both colour
# schemes because EDS typography is scheme-independent.
TYPOGRAPHY = Typography(
    title=16,
    nav_label=14,
    menu_item=12,
)


def palette(color_scheme: ColorScheme) -> dict[str, str]:
    """Return the flat ``variable`` -> value map for ``color_scheme``.

    These are the semantic design tokens referenced by name from the
    ``theme.qss.in`` template. Colour variables are resolved from the bundled
    EDS semantic tokens (via :data:`_SEMANTIC_MAP`); typography variables carry
    scheme-independent pixel sizes. The template is scheme-independent - each
    scheme plugs in its own values here, mirroring how a CSS ``:root`` block
    overrides custom properties per theme.

    Args:
        color_scheme: The scheme whose values should be resolved.

    Returns:
        A mapping from template variable name (without the leading ``@``) to the
        concrete QSS value (an ``#rrggbb`` colour or a ``<n>px`` size).
    """
    values = {
        variable: semantic(color_scheme, token)
        for variable, token in _SEMANTIC_MAP.items()
    }
    values["font-title"] = f"{TYPOGRAPHY.title}px"
    values["font-nav-label"] = f"{TYPOGRAPHY.nav_label}px"
    values["font-menu-item"] = f"{TYPOGRAPHY.menu_item}px"
    return values
