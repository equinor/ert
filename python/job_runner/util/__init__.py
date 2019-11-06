import sys
import os

REQUESTED_HEXVERSION = 0x02070000


def check_version():
    if sys.hexversion < REQUESTED_HEXVERSION:
        version = sys.version_info
        warning = """
/------------------------------------------------------------------------
| You are running Python version {}.{}.{}; much of the ert functionality
| expects to be running on Python 2.7.5.  Version 2.7.13 is the default
| version in /prog/sdpsoft.
|
| It is highly recommended that you update your setup to use Python 2.7.5.
|
\\------------------------------------------------------------------------

""".format(
            version[0], version[1], version[2]
        )
        return warning
    return None


def read_os_release(pfx="LSB_"):
    fname = "/etc/os-release"
    if not os.path.isfile(fname):
        return {}

    def processline(ln):
        return ln.strip().replace('"', "")

    def splitline(ln, pfx=""):
        if ln.count("=") == 1:
            k, v = ln.split("=")
            return pfx + k, v
        return None

    props = {}
    with open(fname, "r") as f:
        for line in f:
            kv = splitline(processline(line), pfx=pfx)
            if kv:
                props[kv[0]] = kv[1]
    return props


def pad_nonexisting(path, pad="-- "):
    return path if os.path.exists(path) else pad + path
