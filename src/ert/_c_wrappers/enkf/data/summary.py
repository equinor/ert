from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


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
        super().__init__(c_ptr)
        self._undefined_value = self._get_undef_value()

    def __len__(self):
        return self._length()

    def __repr__(self):
        return f"Summary(key={self.key}, length={len(self)}) {self._ad_str()}"

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
            raise IndexError(f"Invalid index:{index}  Valid range: [0,{len(self)}>")

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
