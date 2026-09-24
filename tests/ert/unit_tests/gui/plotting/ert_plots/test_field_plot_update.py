from unittest.mock import Mock

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ert.config import Field
from ert.field_utils import ErtboxParameters, FieldFileFormat
from ert.gui.plotting.ert_plots.field_update_plot import FieldUpdatePlot
from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils import PlotContext


def test_that_field_update_shows_posterior_mean_minus_prior_mean() -> None:
    prior = EnsembleObject("prior", "prior-id", False, "experiment", "")
    posterior = EnsembleObject("posterior", "posterior-id", False, "experiment", "")
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
        figure.axes[0].images[0].get_array(), np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    assert figure.axes[0].images[0].norm.vmin == -4.0
    assert figure.axes[0].images[0].norm.vmax == 4.0


def test_that_field_update_requires_exactly_two_ensembles() -> None:
    plot_context = Mock(spec=PlotContext)
    plot_context.ensembles.return_value = []
    figure = Figure()

    FieldUpdatePlot().plot(figure, plot_context, {}, pd.DataFrame(), {}, None)

    assert figure.axes[0].texts[0].get_text() == "Select exactly two ensembles"
