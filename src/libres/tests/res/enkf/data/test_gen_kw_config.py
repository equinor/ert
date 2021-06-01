from ecl.util.test import TestAreaContext
from tests import ResTest
from res.test import ErtTestContext
from tests.utils import tmpdir
from res.enkf import ErtImplType, GenKwConfig


class GenKwConfigTest(ResTest):
    def test_gen_kw_config(self):
        with TestAreaContext("gen_kw_config"):
            with open("template.txt", "w") as f:
                f.write("Hello")

            with open("parameters.txt", "w") as f:
                f.write("KEY  UNIFORM 0 1 \n")

            with open("parameters_with_comments.txt", "w") as f:
                f.write("KEY1  UNIFORM 0 1 -- COMMENT\n")
                f.write("\n\n")  # Two blank lines
                f.write("KEY2  UNIFORM 0 1\n")
                f.write("--KEY3  \n")
                f.write("KEY3  UNIFORM 0 1\n")

            template_file = "template.txt"
            parameter_file = "parameters.txt"
            parameter_file_comments = "parameters_with_comments.txt"
            with self.assertRaises(IOError):
                conf = GenKwConfig("KEY", template_file, "does_not_exist")

            with self.assertRaises(IOError):
                conf = GenKwConfig("Key", "does_not_exist", parameter_file)

            conf = GenKwConfig("KEY", template_file, parameter_file)
            conf = GenKwConfig("KEY", template_file, parameter_file_comments)
            self.assertEqual(len(conf), 3)

    @tmpdir(None)
    def test_gen_kw_config_get_priors(self):
        parameter_file = "parameters.txt"
        template_file = "template.txt"

        with open(template_file, "w") as f:
            f.write("Hello")

        with open(parameter_file, "w") as f:
            f.write("KEY1  NORMAL 0 1\n")
            f.write("KEY2  LOGNORMAL 2 3\n")
            f.write("KEY3  TRUNCATED_NORMAL 4 5 6 7\n")
            f.write("KEY4  TRIANGULAR 0 1 2\n")
            f.write("KEY5  UNIFORM 2 3\n")
            f.write("KEY6  DUNIF 3 0 1\n")
            f.write("KEY7  ERRF 0 1 2 3\n")
            f.write("KEY8  DERRF 0 1 2 3 4\n")
            f.write("KEY9  LOGUNIF 0 1\n")
            f.write("KEY10  CONST 10\n")

        conf = GenKwConfig("KEY", template_file, parameter_file)
        priors = conf.get_priors()
        self.assertEqual(len(conf), 10)

        assert {
            "key": "KEY1",
            "function": "NORMAL",
            "parameters": {"MEAN": 0, "STD": 1},
        } in priors

        assert {
            "key": "KEY2",
            "function": "LOGNORMAL",
            "parameters": {"MEAN": 2, "STD": 3},
        } in priors

        assert {
            "key": "KEY3",
            "function": "TRUNCATED_NORMAL",
            "parameters": {"MEAN": 4, "STD": 5, "MIN": 6, "MAX": 7},
        } in priors

        assert {
            "key": "KEY4",
            "function": "TRIANGULAR",
            "parameters": {"XMIN": 0, "XMODE": 1, "XMAX": 2},
        } in priors

        assert {
            "key": "KEY5",
            "function": "UNIFORM",
            "parameters": {"MIN": 2, "MAX": 3},
        } in priors

        assert {
            "key": "KEY6",
            "function": "DUNIF",
            "parameters": {"STEPS": 3, "MIN": 0, "MAX": 1},
        } in priors

        assert {
            "key": "KEY7",
            "function": "ERRF",
            "parameters": {"MIN": 0, "MAX": 1, "SKEWNESS": 2, "WIDTH": 3},
        } in priors

        assert {
            "key": "KEY8",
            "function": "DERRF",
            "parameters": {"STEPS": 0, "MIN": 1, "MAX": 2, "SKEWNESS": 3, "WIDTH": 4},
        } in priors

        assert {
            "key": "KEY9",
            "function": "LOGUNIF",
            "parameters": {"MIN": 0, "MAX": 1},
        } in priors

        assert {
            "key": "KEY10",
            "function": "CONST",
            "parameters": {"VALUE": 10},
        } in priors
