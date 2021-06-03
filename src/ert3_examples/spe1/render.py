#!/usr/bin/env python3


# TODO: This script is a hack in its entirety to get hold the template_render
# job from libres. Over time this script should disappear...


import pathlib
import res
import subprocess
import sys


def _locate_template_render_exec():
    return (
        pathlib.Path(res.__file__).parent.parent.parent.parent.parent
        / "share"
        / "ert"
        / "forward-models"
        / "templating"
        / "script"
        / "template_render"
    )


if __name__ == "__main__":
    render_exec = _locate_template_render_exec()
    args = tuple(sys.argv[1:])
    subprocess.run(" ".join((str(render_exec),) + args), shell=True, check=True)
