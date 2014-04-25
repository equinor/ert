#  Copyright (C) 2014  Statoil ASA, Norway.
#
#  The file 'load_results_model.py' is part of ERT - Ensemble based Reservoir Tool.
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
import os

from ert.enkf import EnkfConfigNode, GenKw, EnkfNode, NodeId
from ert_gui.models import ErtConnector

class LoadResultsModel(ErtConnector):

    def __init__(self):
        super(LoadResultsModel, self).__init__()

    def loadResults(self, selected_case, realisations, iterations):
        """
        @type selected_case: str
        @type realisations: BoolVector
        @type iterations: BoolVector
        """
        pass

    def getIterationCount(self):
        run_path = self.ert().getModelConfig().getRunpathAsString()

        formated = run_path % 0, 0

        have_iterations = isinstance(formated, str)

        test = os.path.exists("/")
