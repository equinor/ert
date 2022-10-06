import errno
import os
from typing import TYPE_CHECKING

from cwrap import BaseCClass
from ecl.util.util import CTime

from ert._c_wrappers import ResPrototype
from ert._clib import time_map

if TYPE_CHECKING:
    from ecl.summary import EclSum


class TimeMap(BaseCClass):
    TYPE_NAME = "time_map"

    _alloc = ResPrototype("void*  time_map_alloc()", bind=False)
    _load = ResPrototype("bool   time_map_fread(time_map , char*)")
    _save = ResPrototype("void   time_map_fwrite(time_map , char*)")
    _fload = ResPrototype("bool   time_map_fscanf(time_map , char*)")
    _iget_sim_days = ResPrototype("double time_map_iget_sim_days(time_map, int)")
    _iget = ResPrototype("time_t time_map_iget(time_map, int)")
    _size = ResPrototype("int    time_map_get_size(time_map)")
    _try_update = ResPrototype("bool   time_map_try_update(time_map , int , time_t)")
    _lookup_time = ResPrototype("int    time_map_lookup_time( time_map , time_t)")
    _lookup_time_with_tolerance = ResPrototype(
        "int    time_map_lookup_time_with_tolerance( time_map , time_t , int , int)"
    )
    _lookup_days = ResPrototype(
        "int    time_map_lookup_days( time_map ,         double)"
    )
    _last_step = ResPrototype("int    time_map_get_last_step( time_map )")
    _attach_refcase = ResPrototype("bool time_map_attach_refcase(time_map, ecl_sum)")
    _free = ResPrototype("void   time_map_free( time_map )")

    def __init__(self, filename=None):
        c_ptr = self._alloc()
        super().__init__(c_ptr)
        if filename:
            self.load(filename)

    def load(self, filename):
        if os.path.isfile(filename):
            self._load(filename)
        else:
            raise IOError((errno.ENOENT, f"File not found: {filename}"))

    def fwrite(self, filename):
        self._save(filename)

    def fload(self, filename):
        """
        Will load a timemap as a formatted file consisting of a list of dates:
        YYYY-MM-DD
        """
        if os.path.isfile(filename):
            OK = self._fload(filename)
            if not OK:
                raise Exception(f"Error occured when loading timemap from:{filename}")
        else:
            raise IOError((errno.ENOENT, f"File not found: {filename}"))

    def __eq__(self, other):
        return list(self) == list(other)

    def getSimulationDays(self, step):
        """@rtype: double"""
        if not isinstance(step, int):
            raise TypeError("Expected an integer")

        size = len(self)
        if step < 0 or step >= size:
            raise IndexError(f"Index out of range: 0 <= {step} < {size}")

        return self._iget_sim_days(step)

    def attach_refcase(self, refcase):
        self._attach_refcase(refcase)

    def __getitem__(self, index):
        """@rtype: CTime"""
        if not isinstance(index, int):
            raise TypeError("Expected an integer")

        size = len(self)
        if index < 0 or index >= size:
            raise IndexError(f"Index out of range: 0 <= {index} < {size}")

        return self._iget(index)

    def __setitem__(self, index, time):
        self.update(index, time)

    def update(self, index, time):
        if self._try_update(index, CTime(time)):
            return True
        raise Exception("Tried to update with inconsistent value")

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1

    def __contains__(self, time):
        index = self._lookup_time(CTime(time))
        return index >= 0

    def summary_update(self, summary: "EclSum") -> str:
        return time_map.summary_update(self, summary)

    def lookupTime(self, time, tolerance_seconds_before=0, tolerance_seconds_after=0):
        """Will look up the report step corresponding to input @time.

        If the tolerance arguments tolerance_seconds_before and
        tolerance_seconds_after have the default value zero we require
        an exact match between input time argument and the content of
        the time map.

        If the tolerance arguments are supplied the function will
        search through the time_map for the report step closest to the
        time argument, which satisfies the tolerance criteria.

        With the call:

            lookupTime( datetime.date(2010,1,10) , 3600*24 , 3600*7)

        We will find the report step in the date interval 2010,1,9 -
        2010,1,17 which is closest to 2010,1,10. The tolerance limits
        are inclusive.

        If no report step satisfying the criteria is found a
        ValueError exception will be raised.

        """
        if tolerance_seconds_before == 0 and tolerance_seconds_after == 0:
            index = self._lookup_time(CTime(time))
        else:
            index = self._lookup_time_with_tolerance(
                CTime(time), tolerance_seconds_before, tolerance_seconds_after
            )

        if index >= 0:
            return index
        else:
            raise ValueError(f"The time:{time} was not found in the time_map instance")

    def lookupDays(self, days):
        index = self._lookup_days(days)
        if index >= 0:
            return index
        else:
            raise ValueError(f"The days: {days} was not found in the time_map instance")

    def __len__(self):
        """@rtype: int"""
        return self._size()

    def free(self):
        self._free()

    def __repr__(self):
        return self._create_repr(f"size = {len(self)}")

    def dump(self):
        """
        Will return a list of tuples (step , CTime , days).
        """
        step_list = []
        for step, t in enumerate(self):
            step_list.append((step, t, self.getSimulationDays(step)))
        return step_list
