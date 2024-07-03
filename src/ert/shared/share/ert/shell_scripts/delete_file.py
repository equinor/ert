#!/usr/bin/env python
import os
import sys


def delete_file(filename):
    if os.path.exists(filename):
        if os.path.isfile(filename):
            stat_info = os.stat(filename)
            uid = stat_info.st_uid
            if uid == os.getuid():
                os.unlink(filename)
                print(f"Removing file:'{filename}'")
            else:
                sys.stderr.write(
                    f"Sorry you are not owner of file:{filename} - not deleted\n"
                )
        else:
            raise IOError(f"Entry:'{filename}' is not a regular file")
    elif os.path.islink(filename):
        os.remove(filename)
    else:
        sys.stderr.write(f"File: '{filename}' does not exist - delete ignored\n")


if __name__ == "__main__":
    try:
        for file in sys.argv[1:]:
            delete_file(file)
    except IOError as e:
        sys.exit(f"DELETE_FILE failed with the following error: {e}")
