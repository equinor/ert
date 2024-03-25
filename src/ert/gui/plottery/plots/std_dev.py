import io
import math
from typing import List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ert.gui.plottery import PlotContext


class StdDevPlot:
    def __init__(self):
        self.dimensionality = 3

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map,
        _observation_data,
        std_dev_images: List[bytes],
    ):
        sub_plot_dim = math.ceil(math.sqrt(len(std_dev_images)))
        for i, img in enumerate(
            [plt.imread(io.BytesIO(img)) for img in std_dev_images]
        ):
            ax = figure.add_subplot(sub_plot_dim, sub_plot_dim, i + 1)
            ax.imshow(img)
            ax.axis("off")
        figure.tight_layout()
