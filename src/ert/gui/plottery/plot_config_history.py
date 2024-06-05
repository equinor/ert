from typing import List

from ert.gui.plottery import PlotConfig


class PlotConfigHistory:
    """A Class for tracking changes to a PlotConfig class (supports undo, redo
    and reset)"""

    def __init__(self, name: str, initial: PlotConfig) -> None:
        super().__init__()
        self._name = name
        self._initial = PlotConfig.createCopy(initial)
        self._undo_history: List[PlotConfig] = []
        self._redo_history: List[PlotConfig] = []
        self._current = PlotConfig.createCopy(self._initial)

    def isUndoPossible(self) -> bool:
        return len(self._undo_history) > 0

    def isRedoPossible(self) -> bool:
        return len(self._redo_history) > 0

    def applyChanges(self, plot_config: PlotConfig) -> None:
        self._undo_history.append(self._current)
        copy = PlotConfig.createCopy(self._current)
        copy.copyConfigFrom(plot_config)
        self._current = copy
        del self._redo_history[:]

    def resetChanges(self) -> None:
        self.applyChanges(self._initial)

    def undoChanges(self) -> None:
        if self.isUndoPossible():
            self._redo_history.append(self._current)
            self._current = self._undo_history.pop()

    def redoChanges(self) -> None:
        if self.isRedoPossible():
            self._undo_history.append(self._current)
            self._current = self._redo_history.pop()

    def getPlotConfig(self) -> PlotConfig:
        return PlotConfig.createCopy(self._current)
