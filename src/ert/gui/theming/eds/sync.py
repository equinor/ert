"""Sync the bundled Equinor Design System (EDS) colour tokens.

The ERT GUI mirrors the EDS colour foundation. Rather than hand-copying
hundreds of hex values, this module downloads the *resolved* token JSON that
the ``equinor/design-system`` build pipeline publishes and rewrites the bundled
files consumed by :mod:`ert.gui.theming.tokens`.

Two EDS layers are captured per colour scheme:

* ``scale`` - the generic colour scales (``accent-1``..``danger-15``); these are
  the tier-1 "generic colours".
* ``semantic`` - the purpose tokens (``bg-*``, ``border-*``, ``text-*``) that a
  stylesheet should reference by role rather than by shade.

Run it with::

    uv run python -m ert.gui.theming.eds.sync

then regenerate the stylesheets with::

    uv run python -m ert.gui.theming.generate_qss
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Final

from ert.gui.theming.theme import ColorScheme

from .data import bundled_path

_RAW_BASE: Final = (
    "https://raw.githubusercontent.com/equinor/design-system/main/"
    "packages/eds-tokens/build/json/color/color-scheme/flat"
)

# EDS publishes the resolved scales and the semantic (purpose) tokens as two
# separate flat files per scheme; we merge them into one bundled file.
_LAYERS: Final = {
    "scale": "{scheme}-color-scheme",
    "semantic": "{scheme}-semantic",
}


def _fetch(url: str) -> dict[str, str]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _bundle_for(scheme: ColorScheme) -> dict[str, dict[str, str]]:
    bundle: dict[str, dict[str, str]] = {}
    for layer, name_template in _LAYERS.items():
        name = name_template.format(scheme=scheme.value)
        bundle[layer] = _fetch(f"{_RAW_BASE}/{name}.json")
    return bundle


def sync() -> list[Path]:
    """Download and rewrite the bundled EDS token files.

    Returns:
        The bundled file paths that were written.
    """
    written: list[Path] = []
    for scheme in ColorScheme:
        path = bundled_path(scheme)
        path.write_text(
            json.dumps(_bundle_for(scheme), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


if __name__ == "__main__":
    for written_path in sync():
        print(f"wrote {written_path}")
