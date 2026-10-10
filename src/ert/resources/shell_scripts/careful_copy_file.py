#!/usr/bin/env python
import shutil
import sys
from pathlib import Path


def careful_copy_file(src: str, target: str | None = None) -> None:
    if target is None:
        target = Path(src).name
    if Path(target).exists():
        print(f"File: {target} already present - not updated")
        return
    if Path(src).is_file():
        if Path(target).is_dir():
            target_file = str(Path(target) / Path(src).name)
            shutil.copyfile(src, target_file)
            print(f"Copying file '{src}' -> '{target_file}'")
        else:
            target_path = Path(target) if target.endswith("/") else Path(target).parent
            if target_path != Path() and not target_path.is_dir():
                target_path.mkdir(parents=True)
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
            careful_copy_file(src, target)
        else:
            careful_copy_file(src)
    except OSError as e:
        sys.exit(f"CAREFUL_COPY_FILE failed with the following error: {e}")
