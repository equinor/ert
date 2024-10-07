import datetime
from copy import copy

from ert.gui.tools.plot.plottery import PlotLimits


def test_plot_limits_construction():
    plot_limits = PlotLimits()
    assert plot_limits.value_minimum is None
    assert plot_limits.value_maximum is None
    assert plot_limits.value_limits == (None, None)

    assert plot_limits.index_minimum is None
    assert plot_limits.index_maximum is None
    assert plot_limits.index_limits == (None, None)

    assert plot_limits.count_minimum is None
    assert plot_limits.count_maximum is None
    assert plot_limits.count_limits == (None, None)

    assert plot_limits.density_minimum is None
    assert plot_limits.density_maximum is None
    assert plot_limits.density_limits == (None, None)

    assert plot_limits.date_minimum is None
    assert plot_limits.date_maximum is None
    assert plot_limits.date_limits == (None, None)


def test_plot_limits():
    plot_limits = PlotLimits()
    limit_names = ["value", "index", "count", "density", "date"]

    positive_floats = [1.0, 1.5, 3.1415, 1e10, 5.2e-7]
    negative_floats = [-1.0, -1.5, -3.1415, -1e10, -5.2e-7]
    positive_ints = [1, 5, 1000]
    negative_ints = [-1, -5, -1000]

    dates = [datetime.date(2001, 1, 1), datetime.date(2010, 3, 3)]

    setter_should_succeed_values = {
        "value": positive_floats + negative_floats + positive_ints + negative_ints,
        "index": positive_ints,
        "count": positive_ints,
        "density": positive_floats + positive_ints,
        "date": dates,
    }

    for attribute_name in limit_names:
        assert getattr(plot_limits, f"{attribute_name}_minimum") is None
        assert getattr(plot_limits, f"{attribute_name}_maximum") is None
        assert getattr(plot_limits, f"{attribute_name}_limits") == (None, None)

        setattr(plot_limits, f"{attribute_name}_minimum", None)
        setattr(plot_limits, f"{attribute_name}_maximum", None)
        setattr(plot_limits, f"{attribute_name}_limits", (None, None))

        for value in setter_should_succeed_values[attribute_name]:
            setattr(plot_limits, f"{attribute_name}_minimum", value)
            setattr(plot_limits, f"{attribute_name}_maximum", value)

            minimum = getattr(plot_limits, f"{attribute_name}_minimum")
            maximum = getattr(plot_limits, f"{attribute_name}_maximum")

            assert minimum == value
            assert maximum == value

            assert getattr(plot_limits, f"{attribute_name}_limits") == (
                minimum,
                maximum,
            )


def test_copy_plot_limits():
    plot_limits = PlotLimits()
    plot_limits.value_limits = 1, 2
    plot_limits.index_limits = 3, 4
    plot_limits.count_limits = 5, 6
    plot_limits.density_limits = 9, 10
    plot_limits.date_limits = (
        datetime.date(1999, 1, 1),
        datetime.date(1999, 12, 31),
    )

    copy_of_plot_limits = copy(plot_limits)

    assert copy_of_plot_limits == plot_limits

    assert copy_of_plot_limits.value_limits == (1, 2)
    assert copy_of_plot_limits.index_limits == (3, 4)
    assert copy_of_plot_limits.count_limits == (5, 6)
    assert copy_of_plot_limits.density_limits == (9, 10)
    assert copy_of_plot_limits.date_limits == (
        datetime.date(1999, 1, 1),
        datetime.date(1999, 12, 31),
    )
