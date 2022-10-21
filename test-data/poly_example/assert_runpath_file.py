#!/usr/bin/env python
import os
import sys


def run():
    runpath_file = sys.argv[1]
    iteration = 0

    ok_filename = "RUNPATH_WORKFLOW_{}.OK"
    if os.path.isfile(ok_filename.format(0)):
        iteration = 1

    curdir = sys.argv[2]
    runpath_line = (
        "{iens:03d}  "
        "{pwd}/poly_out/realization-{iens}/iter-{iter}  "
        "poly_{iens}  {iter:03d}\n"
    )
    with open(runpath_file) as fh:
        rpf = "".join(
            [
                runpath_line.format(iens=iens, iter=iteration, pwd=curdir)
                for iens in [1, 2, 4, 8, 16, 32, 64]
            ]
        )
        assert fh.read() == rpf

    with open(ok_filename.format(iteration), "w") as fh:
        fh.write(":)")


if __name__ == "__main__":
    run()
