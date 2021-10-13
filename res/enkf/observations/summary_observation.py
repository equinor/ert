#  Copyright (C) 2012 Equinor ASA, Norway.
#
#  The file 'summary_observation.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res._clib import _SummaryObservationImpl


class SummaryObservation(_SummaryObservationImpl):
    def __init__(
        self,
        summary_key,
        observation_key,
        value,
        std,
        auto_corrf_name=None,
        auto_corrf_param=0.0,
    ):
        assert isinstance(summary_key, str)
        assert isinstance(observation_key, str)
        assert isinstance(value, float)
        assert isinstance(std, float)

        if auto_corrf_name is not None:
            assert isinstance(auto_corrf_name, str)

        assert isinstance(auto_corrf_param, float)
        super().__init__(summary_key, observation_key, value, std)

    def __len__(self):
        return 1

    def __repr__(self):
        sk = self.getSummaryKey()
        va = self.getValue()
        sd = self.getStandardDeviation()
        sc = self.getStdScaling()
        ad = self._address()
        fmt = "SummaryObservation(key = %s, value = %f, std = %f, std_scaling = %f) at 0x%x"
        return fmt % (sk, va, sd, sc, ad)
