from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.config import Field
from ert.field_utils import ErtboxParameters, FieldFileFormat
from ert.gui.plotting.ert_plots.field_update_plot import FieldUpdatePlot
from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils import PlotContext


def test_that_field_update_shows_posterior_mean_minus_prior_mean():
    # start_at are just dummy values for sorting purposes
    prior = EnsembleObject("prior", "prior-id", False, "experiment", "1")
    posterior = EnsembleObject("posterior", "posterior-id", False, "experiment", "2")
    field = Field(
        name="PORO",
        ertbox_params=ErtboxParameters(2, 2, 1),
        file_format=FieldFileFormat.ROFF_BINARY,
        forward_init=False,
        forward_init_file="init.roff",
        output_file="output.roff",
        grid_file="grid.roff",
        update_strategy="global",
    )
    key_def = PlotApiKeyDefinition(
        key="PORO",
        index_type=None,
        observations=False,
        dimensionality=3,
        metadata={"data_origin": "field"},
        parameter=field,
    )
    plot_context = Mock(spec=PlotContext)
    plot_context.ensembles.return_value = [prior, posterior]
    plot_context.layer = 0
    plot_context.key.return_value = "PORO"

    figure = Figure()
    FieldUpdatePlot().plot(
        figure,
        plot_context,
        {
            prior: pd.DataFrame([1.0, 2.0, 3.0, 4.0]),
            posterior: pd.DataFrame([2.0, 4.0, 6.0, 8.0]),
        },
        pd.DataFrame(),
        {},
        None,
        key_def,
    )

    np.testing.assert_array_equal(
        figure.axes[0].images[0].get_array(), np.array([[1.0, 3.0], [2.0, 4.0]])
    )
    assert figure.axes[0].images[0].norm.vmin == pytest.approx(-4.0)
    assert figure.axes[0].images[0].norm.vmax == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("selected", "expected_hint"),
    [
        (0, "Select two ensembles to be able to make the comparison"),
        (1, "Select an additional ensemble to be able to make the comparison"),
        (3, "Too many ensembles selected (3), select only 2"),
    ],
)
def test_that_field_update_shows_hint_when_not_exactly_two_ensembles_are_selected(
    selected: int, expected_hint: str
):
    plot_context = Mock(spec=PlotContext)
    plot_context.ensembles.return_value = [
        EnsembleObject(f"ens{i}", f"id{i}", False, "experiment", str(i))
        for i in range(selected)
    ]
    figure = Figure()

    FieldUpdatePlot().plot(figure, plot_context, {}, pd.DataFrame(), {}, None)

    assert figure.axes[0].texts[0].get_text() == expected_hint
