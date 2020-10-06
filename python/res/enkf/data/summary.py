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
    _length = ResPrototype("int     summary_length(summary)")

    def __init__(self, config):
        c_ptr = self._alloc(config)
        self._config = config
        super(Summary, self).__init__(c_ptr)

    def __len__(self):
        return self._length()

    def __repr__(self):
        return "Summary(key=%s, length=%d) %s" % (self.key, len(self), self._ad_str())

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0:
            raise ValueError("Invalid index")

        if index >= len(self):
            raise IndexError(
                "Invalid index:%d  Valid range: [0,%d>" % (index, len(self))
            )

        return self._iget_value(index)

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
