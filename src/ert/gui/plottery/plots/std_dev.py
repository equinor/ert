from typing import TYPE_CHECKING

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pandas import DataFrame

from ert.gui.plottery import PlotConfig, PlotContext
from ert.gui.plottery.plots.history import plotHistory

from .observations import plotObservations
from .plot_tools import PlotTools
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

class StdDevPlot:
    def __init__(self):
      self.dimensionality = 2

    def plot(
        self, figure: Figure, plot_context: PlotContext, ensemble_to_data_map, _observation_data
    ):
        pass
        ax = figure.add_subplot(111)

        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox


        arr_img = plt.imread("/home/frode/code/tomt-hytte.png")
        im = OffsetImage(arr_img)
        ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction')
        ax.add_artist(ab)

        PlotTools.finalizePlot(
            plot_context,
            figure,
            ax,
        )

        
