import matplotlib.pyplot as plt
import pandas as pd
import pytest

from fmudesign.quality_report import QualityReporter


@pytest.mark.parametrize("var_name", ["category", "count", "proportion"])
def test_that_discrete_plot_counts_match_sorted_categories(var_name):
    series = pd.Series(["beta", "beta", "beta", "alpha"])

    fig, ax = QualityReporter.plot_discrete(
        series=series,
        var_name=var_name,
        var_description=["Discrete", []],
    )

    assert ax.get_xlabel() == var_name
    categories = [label.get_text() for label in ax.get_xticklabels()]
    annotations = [annotation.get_text() for annotation in ax.texts]
    assert dict(zip(categories, annotations, strict=True)) == {
        "alpha": "25.0% (n=1)",
        "beta": "75.0% (n=3)",
    }

    plt.close(fig)
