from typing import TYPE_CHECKING
import io
import math
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pandas import DataFrame

from ert.gui.plottery import PlotConfig, PlotContext
from ert.gui.plottery.plots.history import plotHistory

from .observations import plotObservations
from .plot_tools import PlotTools
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import List
    

class StdDevPlot:
    def __init__(self):
      self.dimensionality = 2

    def plot(
        self, figure: Figure, plot_context: PlotContext, ensemble_to_data_map, _observation_data, std_dev_images : List[bytes]
    ):
        sub_plot_dim = math.ceil(math.sqrt(len(std_dev_images)))
        for i, img in enumerate([plt.imread(io.BytesIO(img)) for img in std_dev_images]):
            ax = figure.add_subplot(sub_plot_dim, sub_plot_dim, i+1)
            ax.imshow(img)
            ax.axis("off")
        figure.tight_layout()
       
