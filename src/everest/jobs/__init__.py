import importlib
import os
from pathlib import Path

script_names = ("render",)

__all__ = [
    "script_names",
]


def _inject_scripts() -> None:
    def fetch_script_path(script_name: str) -> str:
        rel_script_path = os.path.join("scripts", script_name)
        return str(
            Path(importlib.util.find_spec("everest.jobs").origin).parent  # type: ignore
            / rel_script_path
        )

    scripts = {}
    for script_name in script_names:
        scripts[script_name] = fetch_script_path(script_name)
        globals()[script_name] = scripts[script_name]

    globals()["_scripts"] = scripts


def fetch_script(script_name: str) -> str:
    if script_name in _scripts:  # type: ignore # noqa F821
        return _scripts[script_name]  # type: ignore # noqa F821
    else:
        raise KeyError(f"Unknown script: {script_name}")


_inject_scripts()
