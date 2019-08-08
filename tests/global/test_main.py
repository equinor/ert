import argparse
import sys
import unittest

from ert_gui.main import ert_parser, main

if sys.version_info >= (3, 3):
    from unittest.mock import Mock, patch
else:
    from mock import Mock, patch


class MainTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MainTest, self).__init__(*args, **kwargs)

        # argv is inconsequential for this case, as parsing is stubbed out.
        # Tests still needs to assert argv was passed, and since it may vary
        # (based on environment) it is unset.
        sys.argv = list()

        # main is essentially a call to ArgumentParser.parse_args, so it is mocked.
        self.arg_parser_mock = None

        # ArgumentParser.parse_args returns a Namespace, which must be mocked as
        # parse_args is stubbed out.
        self.namespace_mock = None

    def mock_parser(self):
        """Returns a mocked ArgumentParser."""
        return self.arg_parser_mock

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.namespace_mock = Mock(spec=argparse.Namespace)
        self.namespace_mock.func = Mock()

        self.arg_parser_mock = Mock(spec=argparse.ArgumentParser)
        self.arg_parser_mock.parse_args = Mock(return_value=self.namespace_mock)

    def test_run_gui(self):
        self.namespace_mock.interface = 'gui'

        with patch('ert_gui.main.ert_parser', new=self.mock_parser):
            main()

        self.arg_parser_mock.parse_args.assert_called_once_with(sys.argv)
        self.namespace_mock.func.assert_called_once_with(self.namespace_mock)

    def test_run_cli_with_valid_arguments(self):
        self.namespace_mock.interface = 'cli'
        self.namespace_mock.mode = 'ensemble_experiment'
        self.namespace_mock.target_case = 'some case'

        with patch('ert_gui.main.ert_parser', new=self.mock_parser):
            main()

        self.arg_parser_mock.parse_args.assert_called_once_with(sys.argv)
        self.namespace_mock.func.assert_called_once_with(self.namespace_mock)

    def test_run_cli_with_invalid_arguments(self):
        self.namespace_mock.interface = 'cli'
        self.namespace_mock.mode = 'ensemble_smoother'
        self.namespace_mock.target_case = 'default'
        self.arg_parser_mock.error = Mock()

        with patch('ert_gui.main.ert_parser', new=self.mock_parser):
            main()

        self.arg_parser_mock.parse_args.assert_called_once_with(sys.argv)
        self.arg_parser_mock.error.assert_called_once_with("Target file system and source file system can not be the same. "
                                                           "They were both: <default>. Please set using --target-case on "
                                                           "the command line.")
        self.namespace_mock.func.assert_called_once_with(self.namespace_mock)


if __name__ == '__main__':
    unittest.main()
