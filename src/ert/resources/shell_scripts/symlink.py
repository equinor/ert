#!/usr/bin/env python
import os
import sys


def symlink(target: str, link_name: str) -> None:
    """Will create a symbol link 'link_name -> target'.

    If the @link_name already exists as a symbolic link it will be
    removed first; if the @link_name exists and is *not* a
    symbolic link OSError will be raised. If the @target does not
    exists IOError will be raised.
    """
    link_path, _ = os.path.split(link_name)
    if len(link_path) == 0:
        target_check = target
    else:
        if not os.path.isdir(link_path):
            print(f"Creating directory for link: {link_path}")
            os.makedirs(link_path)
        target_check = os.path.join(link_path, target)

    if not os.path.exists(target_check):
        raise OSError(
            f"{target} (target) and {link_name} (link_name) requested, "
            f"which implies that {target_check} must exist, but it does not."
        )

    if os.path.islink(link_name):
        os.unlink(link_name)
    os.symlink(target, link_name)
    print(f"Linking '{link_name}' -> '{target}' [ cwd:{os.getcwd()} ]")


if __name__ == "__main__":
    target = sys.argv[1]
    link_name = sys.argv[2]
    try:
        symlink(target, link_name)
    except OSError as e:
        sys.exit(f"SYMLINK failed with the following error: {e}")
