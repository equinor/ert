#!/usr/bin/env python
import os
import sys


def mkdir(path: str) -> None:
    if os.path.isdir(path):
        print(f"OK - directory: '{path}' already exists")
    else:
        try:
            os.makedirs(path)
            print(f"Created directory: '{path}'")
        except OSError as error:
            # Seems in many cases the directory just suddenly appears;
            # synchronization issues?
            if not os.path.isdir(path):
                msg = f'ERROR: Failed to create directory "{path}": {error}.'
                raise OSError(msg) from error


if __name__ == "__main__":
    path = sys.argv[1]
    try:
        mkdir(path)
    except OSError as e:
        sys.exit(f"MAKE_DIRECTORY failed with the following error: {e}")
