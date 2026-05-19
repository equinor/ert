import os
from pathlib import Path


def check_executable(fname: str | None) -> str:
    """The function returns an error message if the given file is either not a file or
    not an executable.

    If the given file name is an absolute path, its functionality is straight
    forward. When given a relative path it will look for the given file in the
    current directory as well as all locations specified by the environment
    path.

    """
    if not fname:
        return "No executable provided!"
    filepath = Path(fname).expanduser()

    potential_executables = [filepath.resolve()]
    if not filepath.is_absolute():
        potential_executables += [
            Path(location) / filepath
            for location in os.environ["PATH"].split(os.pathsep)
        ]

    if not any(path.is_file() for path in potential_executables):
        return f"{filepath} is not a file!"

    if not any(os.access(path, os.X_OK) for path in potential_executables):
        return f"{filepath} is not an executable!"
    return ""
