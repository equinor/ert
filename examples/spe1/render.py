#!/usr/bin/env python
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


render_exec = _locate_template_render_exec()
args = tuple(sys.argv[1:])
subprocess.run(" ".join((str(render_exec),) + args), shell=True, check=True)
