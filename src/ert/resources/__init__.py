import os
from pathlib import Path

all_shell_script_fm_steps = [
    x.split(".py")[0]
    for x in os.listdir(Path(__file__).resolve().parent / "shell_scripts")
    if not x.startswith("__")
]

__all__ = ["all_shell_script_fm_steps"]
