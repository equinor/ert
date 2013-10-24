from ert_gui.ide.keywords.data import Argument, Keyword, ValidationStatus
from ert_gui.ide.keywords.definitions import ArgumentDefinition, KeywordDefinition


class ConfigurationLine(object):
    ARGUMENT_NOT_EXPECTED = "Argument not expected!"
    ARGUMENT_ERROR = "Keyword has an argument error!"
    UNKNOWN_KEYWORD = "Unknown keyword!"

    def __init__(self, keyword, arguments):
        super(ConfigurationLine, self).__init__()

        #: :type: Keyword
        self.__keyword = keyword
        self.__arguments = []
        keyword.setValidationStatus(ValidationStatus())

        if keyword.hasKeywordDefinition():
            #: :type: list of ArgumentDefinition
            arg_defs = keyword.keywordDefinition().arguments

            arg_def_count = len(arg_defs)
            arg_count = len(arguments)

            #todo check if last argument is optional....


            if arg_count > arg_def_count:
                # merge last input arguments

                last_arg_def = arg_defs[len(arg_defs) - 1]

                if last_arg_def.consumeRestOfLine():
                    from_arg = arguments[arg_def_count - 1]
                    to_arg = arguments[arg_count - 1]

                    last_argument = Argument(from_arg.fromIndex(), to_arg.toIndex(), from_arg.line())
                    arguments = arguments[0:arg_def_count]
                    arguments[len(arguments) - 1] = last_argument

                else:
                    from_arg = arguments[arg_def_count]
                    to_arg = arguments[arg_count - 1]

                    last_argument = Argument(from_arg.fromIndex(), to_arg.toIndex(), from_arg.line())
                    arguments = arguments[0:arg_def_count]
                    arguments.append(last_argument)

            if arg_count < arg_def_count:
                # pad with empty arguments
                line = keyword.line()

                for index in range(arg_def_count - arg_count):
                    empty_argument = Argument(len(line), len(line), line)
                    arguments.append(empty_argument)


            failed_argument = False
            for index in range(len(arguments)):

                arg = arguments[index]

                if index < len(arg_defs):
                    arg_def = arg_defs[index]
                    validation_status = arg_def.validate(arg.value())

                    arg.setValidationStatus(validation_status)
                    arg.setArgumentType(arg_def)
                else:
                    validation_status = ValidationStatus()
                    validation_status.setFailed()
                    validation_status.addToMessage(ConfigurationLine.ARGUMENT_NOT_EXPECTED)
                    arg.setValidationStatus(validation_status)

                if not arg.validationStatus():
                    failed_argument = True

                self.__arguments.append(arg)

            if failed_argument:
                keyword.validationStatus().setFailed()
                keyword.validationStatus().addToMessage(ConfigurationLine.ARGUMENT_ERROR)

        else:
            keyword.validationStatus().setFailed()
            keyword.validationStatus().addToMessage(ConfigurationLine.UNKNOWN_KEYWORD)

            self.__arguments = arguments

            for argument in arguments:
                status = ValidationStatus()
                status.setFailed()
                status.addToMessage(ConfigurationLine.ARGUMENT_NOT_EXPECTED)
                argument.setValidationStatus(status)


    def keyword(self):
        """ @rtype: Keyword"""
        return self.__keyword

    def arguments(self):
        """ @rtype: list of Argument """
        return self.__arguments

