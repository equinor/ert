# Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'enkf_obs.py' is part of ERT - Ensemble based Reservoir Tool.
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
from cwrap import BaseCClass
from ecl.util.util import StringList

from res import ResPrototype
from res.enkf.enums import EnkfObservationImplementationType
from res.enkf.observations import ObsVector


class EnkfObs(BaseCClass):
    TYPE_NAME = "enkf_obs"

    _get_size = ResPrototype("int enkf_obs_get_size( enkf_obs )")
    _valid = ResPrototype("bool enkf_obs_is_valid(enkf_obs)")

    _clear = ResPrototype("void enkf_obs_clear( enkf_obs )")
    _alloc_typed_keylist = ResPrototype(
        "stringlist_obj enkf_obs_alloc_typed_keylist(enkf_obs, enkf_obs_impl_type)"
    )
    _alloc_matching_keylist = ResPrototype(
        "stringlist_obj enkf_obs_alloc_matching_keylist(enkf_obs, char*)"
    )
    _has_key = ResPrototype("bool enkf_obs_has_key(enkf_obs, char*)")
    _obs_type = ResPrototype("enkf_obs_impl_type enkf_obs_get_type(enkf_obs, char*)")
    _get_vector = ResPrototype("obs_vector_ref enkf_obs_get_vector(enkf_obs, char*)")
    _iget_vector = ResPrototype("obs_vector_ref enkf_obs_iget_vector(enkf_obs, int)")
    _iget_obs_time = ResPrototype("time_t enkf_obs_iget_obs_time(enkf_obs, int)")
    _add_obs_vector = ResPrototype("void enkf_obs_add_obs_vector(enkf_obs, obs_vector)")

    def __len__(self):
        return self._get_size()

    def __contains__(self, key):
        return self._has_key(key)

    def __iter__(self):
        """@rtype: ObsVector"""
        iobs = 0
        while iobs < len(self):
            vector = self[iobs]
            yield vector
            iobs += 1

    def __getitem__(self, key_or_index):
        """@rtype: ObsVector"""
        if isinstance(key_or_index, str):
            if self.hasKey(key_or_index):
                return self._get_vector(key_or_index).setParent(self)
            else:
                raise KeyError("Unknown key: %s" % key_or_index)
        elif isinstance(key_or_index, int):
            idx = key_or_index
            if idx < 0:
                idx += len(self)
            if 0 <= idx < len(self):
                return self._iget_vector(idx).setParent(self)
            else:
                raise IndexError(
                    "Invalid index: %d.  Valid range is [0, %d)."
                    % (key_or_index, len(self))
                )
        else:
            raise TypeError(
                "Key or index must be of type str or int, not %s."
                % str(type(key_or_index))
            )

    def getTypedKeylist(
        self, observation_implementation_type: EnkfObservationImplementationType
    ) -> StringList:
        """
        @type observation_implementation_type: EnkfObservationImplementationType
        @rtype: StringList
        """
        return self._alloc_typed_keylist(observation_implementation_type)

    def obsType(self, key):
        if key in self:
            return self._obs_type(key)
        else:
            raise KeyError("Unknown observation key:%s" % key)

    def getMatchingKeys(self, pattern, obs_type=None):
        """
        Will return a list of all the observation keys matching the input
        pattern. The matching is based on fnmatch().
        """
        key_list = self._alloc_matching_keylist(pattern)
        if obs_type:
            new_key_list = []
            for key in key_list:
                if self.obsType(key) == obs_type:
                    new_key_list.append(key)
            return new_key_list
        else:
            return key_list

    def hasKey(self, key):
        """@rtype: bool"""
        return key in self

    def getObservationTime(self, index):
        """@rtype: CTime"""
        return self._iget_obs_time(index)

    def addObservationVector(self, observation_vector):
        assert isinstance(observation_vector, ObsVector)

        observation_vector.convertToCReference(self)

        self._add_obs_vector(observation_vector)

    @property
    def valid(self):
        return self._valid()

    def clear(self):
        self._clear()

    def __repr__(self):
        validity = "valid" if self.valid else "invalid"
        return self._create_repr("%s, len=%d" % (validity, len(self)))
