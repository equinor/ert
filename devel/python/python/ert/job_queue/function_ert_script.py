from ert.cwrap import clib, CWrapper
from ert.job_queue import ErtScript


class FunctionErtScript(ErtScript):

    def __init__(self, ert, function_name, argument_types):
        super(FunctionErtScript, self).__init__(ert)

        lib = clib.ert_load(None)
        wrapper = CWrapper(lib)

        parsed_argument_types = []
        for arg in argument_types:
            if arg is bool:
                parsed_argument_types.append("bool")
            elif arg is str:
                parsed_argument_types.append("char*")
            elif arg is int:
                parsed_argument_types.append("int")
            elif arg is float:
                parsed_argument_types.append("float")
            else:
                raise TypeError("Unknown type: %s" % arg)

        self.__function = wrapper.prototype("c_void_p %s(%s)" % (function_name, ", ".join(parsed_argument_types)))


    def run(self, *args):
        return self.__function(*args)

    def cancel(self):
        # super(FunctionErtScript, self).cancel()
        print("Unable to cancel this type of job! You just have to wait!")




