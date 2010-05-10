
class enum:
    _enums = {}  #This contains all sub classed enums! {class : [list of enums], ...}
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


class ert_job_status_type(enum):
    """These "enum" values are all copies from the header file "basic_queue_driver.h"."""
    # Observe that the status strings are available from the function: libjob_queue.job_queue_status_name( status_code )
    NOT_ACTIVE  = None
    LOADING     = None
    WAITING     = None
    PENDING     = None
    RUNNING     = None
    DONE        = None
    EXIT        = None
    RUN_OK      = None
    RUN_FAIL    = None
    ALL_OK      = None
    ALL_FAIL    = None
    USER_KILLED = None
    USER_EXIT   = None

ert_job_status_type.NOT_ACTIVE = ert_job_status_type("JOB_QUEUE_NOT_ACTIVE", 1)
ert_job_status_type.LOADING = ert_job_status_type("JOB_QUEUE_LOADING", 2)
ert_job_status_type.WAITING = ert_job_status_type("JOB_QUEUE_WAITING", 4)
ert_job_status_type.PENDING = ert_job_status_type("JOB_QUEUE_PENDING", 8)
ert_job_status_type.RUNNING = ert_job_status_type("JOB_QUEUE_RUNNING", 16)
ert_job_status_type.DONE = ert_job_status_type("JOB_QUEUE_DONE", 32)
ert_job_status_type.EXIT = ert_job_status_type("JOB_QUEUE_EXIT", 64)
ert_job_status_type.RUN_OK = ert_job_status_type("JOB_QUEUE_RUN_OK", 128)
ert_job_status_type.RUN_FAIL = ert_job_status_type("JOB_QUEUE_RUN_FAIL", 256)
ert_job_status_type.ALL_OK = ert_job_status_type("JOB_QUEUE_ALL_OK", 512)
ert_job_status_type.ALL_FAIL = ert_job_status_type("JOB_QUEUE_ALL_FAIL", 1024)
ert_job_status_type.USER_KILLED = ert_job_status_type("JOB_QUEUE_USER_KILLED", 2048)
ert_job_status_type.USER_EXIT = ert_job_status_type("JOB_QUEUE_USER_EXIT", 4096)


#print enum._enums