from pathlib import Path

all_shell_script_fm_steps = [
    path.stem
    for path in (Path(__file__).resolve().parent / "shell_scripts").glob("*.py")
    if not path.name.startswith("__")
]

__all__ = ["all_shell_script_fm_steps"]
