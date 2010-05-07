
class enum:
    _enums = {}
    def __init__(self, name, value):
        self.name = name
        self.value = value

        if not enum._enums.has_key(self.__class__):
            enum._enums[self.__class__] = []

        enum._enums[self.__class__].append(self)

    @classmethod
    def values(cls):
        """Returns a list of the created enums for a class."""
        return enum._enums[cls]

    @classmethod
    def resolveName(cls, name):
        """Finds an enum based on name. Ignores the case of the name."""
        for e in enum._enums[cls]:
            if e.name.lower() == name.lower():
                return e
        return None

    @classmethod
    def resolveValue(cls, value):
        """If several enums have the same value the first will be returned"""
        for e in enum._enums[cls]:
            if e.value == value:
                return e
        return None

    def __str__(self):
        return self.name

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        if isinstance(other, long) or isinstance(other, int):
            return self.value == other
        else:
            return self.value == other.value

    def __hash__(self):
        return hash("%s : %i" % (self.name, self.value))


class ert_state_enum(enum):
    """Defined in enkf_types.h"""
    FORECAST=None
    ANALYZED=None
    BOTH=None

#ert_state_enum.UNDEFINED = ert_state_enum("Undefined", 0)
#ert_state_enum.SERIALIZED = ert_state_enum("Serialized", 1)
ert_state_enum.FORECAST = ert_state_enum("Forecast", 2)
ert_state_enum.ANALYZED = ert_state_enum("Analyzed", 4)
ert_state_enum.BOTH = ert_state_enum("Both", 6)


class enkf_impl_type(enum):
    """Defined in enkf_types.h"""
    #INVALID = 0
    #IMPL_TYPE_OFFSET = 100
    #STATIC = 100
    FIELD = None
    GEN_KW = None
    SUMMARY = None
    GEN_DATA = None
    #MAX_IMPL_TYPE = 113  #! not good to have several with same value, resolveValue fails!!!

enkf_impl_type.FIELD = enkf_impl_type("Field", 104)
enkf_impl_type.GEN_KW = enkf_impl_type("Keyword", 107)
enkf_impl_type.SUMMARY = enkf_impl_type("Summary", 110)
enkf_impl_type.GEN_DATA = enkf_impl_type("Data", 113)


#print enum._enums