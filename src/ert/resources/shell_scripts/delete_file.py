#!/usr/bin/env python
import os
import sys
from pathlib import Path


def delete_file(filename: str) -> None:
    filepath = Path(filename)
    if filepath.exists():
        if filepath.is_file():
            uid = filepath.stat().st_uid
            if uid == os.getuid():
                filepath.unlink()
                print(f"Removing file:'{filename}'")
            else:
                sys.stderr.write(
                    f"Sorry you are not owner of file:{filename} - not deleted\n"
                )
        else:
            raise OSError(f"Entry:'{filename}' is not a regular file")
    elif filepath.is_symlink():
        filepath.unlink()
    else:
        sys.stderr.write(f"File: '{filename}' does not exist - delete ignored\n")


if __name__ == "__main__":
    try:
        for file in sys.argv[1:]:
            delete_file(file)
    except OSError as e:
        sys.exit(f"DELETE_FILE failed with the following error: {e}")
