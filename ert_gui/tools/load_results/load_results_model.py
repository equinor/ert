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

from ert_shared import ERT


class LoadResultsModel(object):

    @staticmethod
    def loadResults(selected_case, realisations, iteration):
        """
        @type selected_case: str
        @type realisations: BoolVector
        @type iteration: int
        @rtype int: number of loaded realisations
        """
        return ERT.enkf_facade.load_results(selected_case, realisations, iteration)

    @staticmethod
    def isValidRunPath():
        """ @rtype: bool """
        run_path = ERT.enkf_facade.get_runpath_as_string()
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
    def getCurrentRunPath():
        """ @rtype: str """
        return ERT.enkf_facade.get_runpath_as_string()


    @staticmethod
    def getIterationCount():
        """ @rtype: int """
        run_path = ERT.enkf_facade.get_runpath_as_string()
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


