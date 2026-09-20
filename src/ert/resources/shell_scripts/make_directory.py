#!/usr/bin/env python
import sys
from pathlib import Path


def mkdir(path: str) -> None:
    if Path(path).is_dir():
        print(f"OK - directory: '{path}' already exists")
    else:
        try:
            Path(path).mkdir(parents=True)
            print(f"Created directory: '{path}'")
        except OSError as error:
            # Seems in many cases the directory just suddenly appears;
            # synchronization issues?
            if not Path(path).is_dir():
                msg = f'ERROR: Failed to create directory "{path}": {error}.'
                raise OSError(msg) from error


if __name__ == "__main__":
    path = sys.argv[1]
    try:
        mkdir(path)
    except OSError as e:
        sys.exit(f"MAKE_DIRECTORY failed with the following error: {e}")
