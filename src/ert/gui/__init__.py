import os

import matplotlib as mpl


def headless() -> bool:
    return "DISPLAY" not in os.environ


if headless():
    mpl.use("Agg")
else:
    mpl.use("QtAgg")
