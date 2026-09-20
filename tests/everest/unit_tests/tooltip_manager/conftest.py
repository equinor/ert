import pytest


@pytest.fixture
def plot_data_1D() -> tuple[list[int], list[str]]:
    return list(range(10)), [str(i) for i in range(10)]


@pytest.fixture
def plot_data_2D() -> tuple[list[list[int]], list[str]]:
    return [list(range(10)) for _ in range(10)], [f"Point {i}" for i in range(10)]
