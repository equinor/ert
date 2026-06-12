from .plot_config import PlotConfig


class PlotConfigHistory:
    """A Class for tracking changes to a PlotConfig class (supports undo, redo
    and reset)
    """

    def __init__(self, name: str, initial: PlotConfig) -> None:
        super().__init__()
        self._name: str = name
        self._initial: PlotConfig = PlotConfig.create_copy(initial)
        self._undo_history: list[PlotConfig] = []
        self._redo_history: list[PlotConfig] = []
        self._current = PlotConfig.create_copy(self._initial)

    def is_undo_possible(self) -> bool:
        return len(self._undo_history) > 0

    def is_redo_possible(self) -> bool:
        return len(self._redo_history) > 0

    def apply_changes(self, plot_config: PlotConfig) -> None:
        self._undo_history.append(self._current)
        copy = PlotConfig.create_copy(self._current)
        copy.copy_config_from(plot_config)
        self._current = copy
        del self._redo_history[:]

    def reset_changes(self) -> None:
        self.apply_changes(self._initial)

    def undo_changes(self) -> None:
        if self.is_undo_possible():
            self._redo_history.append(self._current)
            self._current = self._undo_history.pop()

    def redo_changes(self) -> None:
        if self.is_redo_possible():
            self._undo_history.append(self._current)
            self._current = self._redo_history.pop()

    def get_plot_config(self) -> PlotConfig:
        return PlotConfig.create_copy(self._current)
