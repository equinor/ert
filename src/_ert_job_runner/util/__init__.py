import os
from typing import Dict, Optional, Tuple


def read_os_release(pfx: str = "LSB_") -> Dict[str, str]:
    fname = "/etc/os-release"
    if not os.path.isfile(fname):
        return {}

    def processline(ln: str) -> str:
        return ln.strip().replace('"', "")

    def splitline(ln: str, pfx: str = "") -> Optional[Tuple[str, str]]:
        if ln.count("=") == 1:
            k, v = ln.split("=")
            return pfx + k, v
        return None

    props = {}
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            kv = splitline(processline(line), pfx=pfx)
            if kv:
                props[kv[0]] = kv[1]
    return props


def pad_nonexisting(path: str, pad: str = "-- ") -> str:
    return path if os.path.exists(path) else pad + path
