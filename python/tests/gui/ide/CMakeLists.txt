set(PYTHON_SOURCES
    __init__.py
    test_bool_argument.py
    test_configuration_line.py
    test_configuration_line_builder.py
    test_configuration_line_definition.py
    test_configuration_line_parser.py
    test_ert_keywords.py
    test_float_argument.py
    test_integer_argument.py
    test_path_argument.py
    test_percent_argument.py
    test_proper_name_argument.py
    test_proper_name_format_argument.py
    test_proper_name_format_string_argument.py
    test_range_string_argument.py
    test_tokens.py
)

add_python_package("python.tests.gui.ide"  ${PYTHON_INSTALL_PREFIX}/tests/gui/ide "${PYTHON_SOURCES}" True)

addPythonTest(tests.gui.ide.test_bool_argument.BoolArgumentTest)
addPythonTest(tests.gui.ide.test_integer_argument.IntegerArgumentTest)
addPythonTest(tests.gui.ide.test_float_argument.FloatArgumentTest)
addPythonTest(tests.gui.ide.test_path_argument.PathArgumentTest)
addPythonTest(tests.gui.ide.test_proper_name_argument.ProperNameArgumentTest)
addPythonTest(tests.gui.ide.test_proper_name_format_argument.ProperNameFormatArgumentTest)
addPythonTest(tests.gui.ide.test_proper_name_format_string_argument.ProperNameFormatStringArgumentTest)
addPythonTest(tests.gui.ide.test_range_string_argument.RangeStringArgumentTest)
addPythonTest(tests.gui.ide.test_configuration_line.ConfigurationLineTest)
addPythonTest(tests.gui.ide.test_configuration_line_builder.ConfigurationLineBuilderTest)
addPythonTest(tests.gui.ide.test_configuration_line_definition.ConfigurationLineDefinitionTest)
addPythonTest(tests.gui.ide.test_configuration_line_parser.ConfigurationLineParserTest)
addPythonTest(tests.gui.ide.test_ert_keywords.ErtKeywordTest)
addPythonTest(tests.gui.ide.test_tokens.TokenTest)
addPythonTest(tests.gui.ide.test_percent_argument.PercentArgumentTest)

add_subdirectory(wizards)
