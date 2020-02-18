#!/usr/bin/env python
import sys
import os


def run(runpath_file, config_directory):
    iteration = 0

    ok_filename = "RUNPATH_WORKFLOW_{}.OK"
    if os.path.isfile(ok_filename.format(0)):
        iteration = 1

    runpath_line = "{iens:03d}  {pwd}/poly_example/poly_out/real_{iens}/iter_{iter}  poly_{iens}  {iter:03d}\n"
    with open(runpath_file) as fh:
        rpf = "".join(
            [
                runpath_line.format(iens=iens, iter=iteration, pwd=config_directory)
                for iens in [1, 2, 4, 8, 16, 32, 64]
            ]
        )
        assert fh.read() == rpf

    with open(ok_filename.format(iteration), "w") as fh:
        fh.write(":)")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
