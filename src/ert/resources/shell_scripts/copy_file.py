#!/usr/bin/env python
import os
import shutil
import sys
from pathlib import Path


def copy_file(src: str, target: str | None = None) -> None:
    if Path(src).is_file():
        if target is None:
            target = Path(src).name

        if Path(target).is_dir():
            target_file = str(Path(target) / Path(src).name)
            shutil.copyfile(src, target_file)
            print(f"Copying file '{src}' -> '{target_file}'")
        else:
            target_path = os.path.dirname(target)
            if target_path and not Path(target_path).is_dir():
                Path(target_path).mkdir(parents=True)
                print(f"Creating directory '{target_path}' ")
            if Path(target).is_dir():
                target_file = str(Path(target) / Path(src).name)
            else:
                target_file = target

            print(f"Copying file '{src}' -> '{target_file}'")
            shutil.copyfile(src, target_file)
    else:
        raise OSError(f"Input argument:'{src}' does not correspond to an existing file")


if __name__ == "__main__":
    src = sys.argv[1]
    try:
        if len(sys.argv) > 2:
            target = sys.argv[2]
            copy_file(src, target)
        else:
            copy_file(src)
    except OSError as e:
        sys.exit(f"COPY_FILE failed with the following error: {e}")
