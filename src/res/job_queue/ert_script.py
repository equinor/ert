import inspect
import sys
import traceback

import importlib.util


class ErtScript(object):
    def __init__(self, ert):
        """
        @type ert: EnKFMain
        """
        super(ErtScript, self).__init__()

        if not hasattr(self, "run"):
            raise UserWarning(
                "ErtScript implementations must provide a method run(self, ert, ...)"
            )

        self.__verbose = False
        self.__ert = ert

        self.__is_cancelled = False
        self.__failed = False
        self._stdoutdata = ""
        self._stderrdata = ""

    @property
    def stdoutdata(self):
        """@rtype: str"""
        if isinstance(self._stdoutdata, bytes):
            self._stdoutdata = self._stdoutdata.decode()
        return self._stdoutdata

    @property
    def stderrdata(self):
        """@rtype: str"""
        if isinstance(self._stderrdata, bytes):
            self._stderrdata = self._stderrdata.decode()
        return self._stderrdata

    def isVerbose(self):
        return self.__verbose

    def ert(self):
        """@rtype: res.enkf.EnKFMain"""
        return self.__ert

    def isCancelled(self):
        """@rtype: bool"""
        return self.__is_cancelled

    def hasFailed(self):
        """@rtype: bool"""
        return self.__failed

    def cancel(self):
        self.__is_cancelled = True

    def cleanup(self):
        """
        Override to perform cleanup after a run.
        """
        pass

    def initializeAndRun(self, argument_types, argument_values, verbose=False):
        """
        @type argument_types: list of type
        @type argument_values: list of string
        @type verbose: bool
        @rtype: unknown
        """
        self.__verbose = verbose
        self.__failed = False

        arguments = []
        for index, arg_value in enumerate(argument_values):
            if index < len(argument_types):
                arg_type = argument_types[index]
            else:
                arg_type = str

            if arg_value is not None:
                arguments.append(arg_type(arg_value))
            else:
                arguments.append(None)

        try:
            return self.run(*arguments)
        except AttributeError as e:
            if not hasattr(self, "run"):
                self.__failed = True
                return (
                    "Script '%s' has not implemented a 'run' function"
                    % self.__class__.__name__
                )
            self.outputStackTrace(e)
            return None
        except KeyboardInterrupt:
            return "Script '%s' cancelled (CTRL+C)" % self.__class__.__name__
        except Exception as e:
            self.outputStackTrace(e)
            return None
        finally:
            self.cleanup()

    __module_count = (
        0  # Need to have unique modules in case of identical object naming in scripts
    )

    def outputStackTrace(self, error=None):
        stack_trace = error or "".join(traceback.format_exception(*sys.exc_info()))
        msg = "The script '{}' caused an error while running:\n{}"

        sys.stderr.write(msg.format(self.__class__.__name__, stack_trace))
        self.__failed = True

    @staticmethod
    def loadScriptFromFile(path):
        """@rtype: type ErtScript"""
        try:
            module_name = "ErtScriptModule_%d" % ErtScript.__module_count
            ErtScript.__module_count += 1

            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return ErtScript.__findErtScriptImplementations(module)
        except Exception as e:
            sys.stderr.write("The script '%s' caused an error during load:\n" % path)
            traceback.print_exception(sys.exc_info()[0], sys.exc_info()[1], None)
            return None

    @staticmethod
    def __findErtScriptImplementations(module):
        """@rtype: ErtScript"""
        result = []
        predicate = (
            lambda member: inspect.isclass(member)
            and member.__module__ == module.__name__
        )
        for name, member in inspect.getmembers(module, predicate):
            if ErtScript in inspect.getmro(member):
                result.append(member)

        if len(result) != 1:
            raise UserWarning(
                "Must have (only) one implementation of ErtScript in a module!"
            )

        return result[0]
