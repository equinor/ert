import os


def cond_unlink(file):
    if os.path.exists(file):
        os.unlink(file)


def assert_file_executable(fname):
    """The function raises an IOError if the given file is either not a file or
    not an executable.

    If the given file name is an absolute path, its functionality is straight
    forward. When given a relative path it will look for the given file in the
    current directory as well as all locations specified by the environment
    path.

    """
    if not fname:
        raise IOError("No executable provided!")
    fname = os.path.expanduser(fname)

    potential_executables = [os.path.abspath(fname)]
    if not os.path.isabs(fname):
        potential_executables = potential_executables + [
            os.path.join(location, fname)
            for location in os.environ["PATH"].split(os.pathsep)
        ]

    if not any(map(os.path.isfile, potential_executables)):
        raise IOError(f"{fname} is not a file!")

    if not any(os.access(fn, os.X_OK) for fn in potential_executables):
        raise IOError(f"{fname} is not an executable!")
