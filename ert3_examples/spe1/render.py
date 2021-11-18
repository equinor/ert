#!/usr/bin/env python3


# TODO: This script is a hack in its entirety to get hold the template_render
# job from share/ert/forward-models/templating/script/
# Over time this script should disappear.

import sys
import subprocess
import pkg_resources


if __name__ == "__main__":
    render_exec = pkg_resources.resource_filename(
        "ert_shared", "share/ert/forward-models/templating/script/template_render"
    )
    args = tuple(sys.argv[1:])
    subprocess.run(" ".join((str(render_exec),) + args), shell=True, check=True)
