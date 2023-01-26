from ecl.util.util.stringlist import StringList

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.job_queue import ErtScript


class FunctionErtScript(ErtScript):
    def __init__(self, ert, function_name, argument_types, argument_count):
        super().__init__(ert)
        self._arg_types = argument_types
        self._arg_count = argument_count
        self._func_name = function_name

    def run(self, *args):
        ert = self.ert()
        if ert is not None:
            str_args = StringList()
            for arg in args:
                str_args.append(arg)
            pointer = ert.from_param(ert) if hasattr(ert, "from_param") else ert

            function = ResPrototype(f"void* {self._func_name}(void*, stringlist)")
            return function(pointer, str_args)
        else:
            parsed_argument_types = []
            for arg in self._arg_types[: self._arg_count]:
                if arg in [bool, int, float]:
                    parsed_argument_types.append(arg.__name__)
                elif arg is str:
                    parsed_argument_types.append("char*")
                else:
                    raise TypeError(f"Unknown type: {arg}")
            function = ResPrototype(
                f"void* {self._func_name}(" f"{', '.join(parsed_argument_types)})"
            )
            return function(*args)

    def cancel(self):
        # job is not cancellable and will just ignore the call
        pass
