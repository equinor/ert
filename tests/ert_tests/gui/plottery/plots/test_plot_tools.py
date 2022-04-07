import pandas as pd
import numpy as np
from ert_gui.plottery.plots.plot_tools import index_is_datetime


def test_index_is_datetime_empty():
    df = pd.DataFrame()
    is_datetime = index_is_datetime(data=df)

    assert not is_datetime


def test_index_is_datetime_int():
    ints = np.random.randint(0, 30, size=10)
    df = pd.DataFrame(ints, columns=["Random"], index=ints)
    is_datetime = index_is_datetime(data=df)

    assert not is_datetime


def test_index_is_datetime():
    df = pd.DataFrame(
        np.random.randint(0, 30, size=10),
        columns=["Random"],
        index=pd.date_range("20220101", periods=10),
    )
    is_datetime = index_is_datetime(data=df)

    assert is_datetime
