import os


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
