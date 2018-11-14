#!/usr/bin/env python
import sys
from res.fm.util import build_from_template

if __name__ == "__main__":
    if sys.argv < 1:
        exit(1)
    src = sys.argv[1]
    if len(sys.argv) > 2:
        target = sys.argv[2]
        build_from_template(src, target)
    else:
        build_from_template(src)
