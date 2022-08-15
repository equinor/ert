from ecl.util.util.stringlist import StringList

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.job_queue import ErtScript


class FunctionErtScript(ErtScript):
    def __init__(self, ert, function_name, argument_types, argument_count):
        super().__init__(ert)

        parsed_argument_types = []

        if ert is not None:
            self.__function = ResPrototype(f"void* {function_name}(void*, stringlist)")

        else:
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
                    raise TypeError(f"Unknown type: {arg}")

            self.__function = ResPrototype(
                f"void* {function_name}("
                f"{', '.join(parsed_argument_types[:argument_count])})"
            )

    def run(self, *args):
        ert = self.ert()
        if ert is None:
            # This is usually used for testing purposes without an ert instance
            return self.__function(*args)
        else:
            str_args = StringList()
            for arg in args:
                str_args.append(arg)

            if hasattr(ert, "from_param"):
                pointer = ert.from_param(ert)
            else:
                pointer = ert  # ...

            return self.__function(pointer, str_args)

    def cancel(self):
        # job is not cancellable and will just ignore the call
        pass
