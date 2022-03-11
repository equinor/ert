#  Copyright (C) 2014  Equinor ASA, Norway.
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

from ecl.util.util import BoolVector

from res.enkf import EnKFMain


class LoadResultsModel:
    @staticmethod
    def loadResults(
        ert: EnKFMain, selected_case: str, realisations: BoolVector, iteration: int
    ) -> int:
        fs = ert.getEnkfFsManager().getFileSystem(selected_case)
        return ert.loadFromForwardModel(realisations, iteration, fs)

    @staticmethod
    def isValidRunPath(run_path):
        """@rtype: bool"""
        try:
            result = run_path % (0, 0)
            return True
        except TypeError:
            pass

        try:
            result = run_path % 0
            return True
        except TypeError:
            pass

        return False

    @staticmethod
    def getIterationCount(run_path):
        """@rtype: int"""
        try:
            results = run_path % (0, 0)
        except TypeError:
            return 0

        iteration = 0
        valid_directory = True
        while valid_directory:
            formatted = run_path % (0, iteration + 1)
            valid_directory = os.path.exists(formatted)
            if valid_directory:
                iteration += 1

        return iteration
