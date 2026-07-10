import polars as pl
import pytest


def test_misfit_collector(snake_oil_case_storage, snake_oil_default_storage, snapshot):
    data = snake_oil_default_storage.load_all_misfit_data()
    snapshot.assert_match(data.write_csv(float_precision=8), "misfit_collector.csv")

    with pytest.raises(pl.exceptions.OutOfBoundsError):
        # realization 60:
        _ = data.row(60)
