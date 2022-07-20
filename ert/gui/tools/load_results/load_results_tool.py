#  Copyright (C) 2014  Equinor ASA, Norway.
#
#  The file 'load_results_tool.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from ert.gui.ertwidgets import resourceIcon
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.load_results import LoadResultsPanel


class LoadResultsTool(Tool):
    def __init__(self, facade):
        self.facade = facade
        super().__init__(
            "Load results manually",
            "tools/load_manually",
            resourceIcon("upload.svg"),
        )
        self.__import_widget = None
        self.__dialog = None
        self.setEnabled(self.is_valid_run_path())

    def trigger(self):
        if self.__import_widget is None:
            self.__import_widget = LoadResultsPanel(self.facade)
        self.__dialog = ClosableDialog(
            "Load results manually", self.__import_widget, self.parent()
        )
        self.__import_widget.setCurrectCase()
        self.__dialog.addButton("Load", self.load)
        self.__dialog.exec_()

    def load(self):
        self.__import_widget.load()
        self.__dialog.accept()

    def is_valid_run_path(self) -> bool:
        """A run path is considered valid if we can
        insert realisation and iteration numbers"""
        try:
            # pylint: disable=pointless-statement
            self.facade.run_path % (0, 0)
            return True
        except TypeError:
            try:
                # pylint: disable=pointless-statement
                self.facade.run_path % 0
                return True
            except TypeError:
                return False
