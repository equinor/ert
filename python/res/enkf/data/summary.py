#  Copyright (C) 2018  Equinor ASA, Norway.
#
#  The file 'summary.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res import ResPrototype


class Summary(BaseCClass):
    TYPE_NAME = "summary"
    _alloc = ResPrototype("void*   summary_alloc(summary_config)", bind=False)
    _free = ResPrototype("void    summary_free(summary)")
    _iget_value = ResPrototype("double  summary_get(summary, int)")
    _iset_value = ResPrototype("void    summary_set(summary, int, double)")
    _length = ResPrototype("int     summary_length(summary)")
    _get_undef_value = ResPrototype("double summary_undefined_value()", bind=False)

    def __init__(self, config):
        c_ptr = self._alloc(config)
        self._config = config
        super(Summary, self).__init__(c_ptr)
        self._undefined_value = self._get_undef_value()

    def __len__(self):
        return self._length()

    def __repr__(self):
        return "Summary(key=%s, length=%d) %s" % (self.key, len(self), self._ad_str())

    #  The Summary class is intended to contain results loaded from an Eclipse
    #  formatted summary file, the class has internal functionality for reading
    #  and interpreting the summary files. In addition it has support for random
    #  access to the elements with __getitem__() and __setitem__(). For observe
    #  the following:
    #
    #   1. The index corresponds to the time axis - i.e. report step in eclipse
    #      speak.
    #
    #   2. When using __setitem__() the container will automatically grow to
    #      the required storage. When growing the underlying storage it will be
    #      filled with undefined values, and if you later try to access those
    #      values with __getitem__ you will get a ValueError exception:
    #
    #         summary = Summary( summary_config )
    #         summary[10] = 25
    #         v = summary[0] -> ValueError("Trying access undefined value")
    #
    #      The access of an undefined value is trapped in Python __getitem__()
    #      - i.e. it can be bypassed from C.

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0:
            raise ValueError("Invalid index")

        if index >= len(self):
            raise IndexError(
                "Invalid index:%d  Valid range: [0,%d>" % (index, len(self))
            )

        value = self._iget_value(index)
        if value == self._undefined_value:
            raise ValueError("Trying to access undefined value")
        return value

    def __setitem__(self, index, value):
        if index < 0:
            raise ValueError("Invalid report step")

        self._iset_value(index, value)

    def value(self, report_step):
        return self[report_step]

    @property
    def config(self):
        return self._config

    @property
    def key(self):
        return self.config.key

    def free(self):
        self._free()
