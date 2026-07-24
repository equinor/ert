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
