import unittest

import yaml

from ert3.parameters import translate_, validate, validate_yaml, ValidationError
from tests.utils import tmpdir


class TestParamValidation(unittest.TestCase):

    @tmpdir("test-data/local/poly_project")
    def test_param_validate(self):
        validate_yaml('input/parameters/coeffs.yml')

    @tmpdir("test-data/local/poly_project")
    def test_pop_name(self):
        with open('input/parameters/coeffs.yml') as file:
            conf = yaml.safe_load(file)
            conf[0].pop('name')

        with self.assertRaises(ValidationError):
            validate(conf)

    @tmpdir("test-data/local/poly_project")
    def test_transform_uniform(self):
        snapshot = validate_yaml('input/parameters/coeffs.yml')
        output = translate_(snapshot)

        self.assertEqual("coeffs_a UNIFORM 0 1", output[0])
        self.assertEqual("coeffs_b UNIFORM 0 2", output[1])
        self.assertEqual("coeffs_c UNIFORM 0 5", output[2])


    @tmpdir("test-data/local/poly_project")
    def test_transform_const(self):
        conf = [{
            'name': 'coeffs',
            'type': 'array',
            'variables': [{
                'name': 'a',
                'distribution': 'const',
                'value': 0.5
            }, {
                'name': 'b',
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            }]
        }]

        snapshot = validate(conf)
        output = translate_(snapshot)

        self.assertEqual('coeffs_a CONST 0.5', output[0])
        self.assertEqual('coeffs_b UNIFORM 0 1', output[1])

    def test_valid_uniform(self):
        conf = [{
            'name': 'coeffs',
            'type': 'array',
            'variables': [{
                'name': 'a',
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            }]
        }]

        validate(conf)

    def test_invalid_uniform(self):
        conf = [{
            'name': 'coeffs',
            'type': 'array',
            'variables': [{
                'name': 'a',
                'distribution': 'uniform',
                'value': 0.5,
                'min': 0
            }]
        }]

        with self.assertRaises(ValidationError):
            validate(conf)

    def test_valid_const(self):
        conf = [{
            'name': 'coeffs',
            'type': 'array',
            'variables': [{
                'name': 'a',
                'distribution': 'const',
                'value': 0.5
            }]
        }]

        validate(conf)

    def test_invalid_const(self):
        conf = [{
            'name': 'coeffs',
            'type': 'array',
            'variables': [{
                'name': 'a',
                'distribution': 'const',
                'value': 0.5,
                'min': 0
            }]
        }]

        with self.assertRaises(ValidationError):
            validate(conf)

    def test_invalid_distribution(self):
        conf = [{
            'name': 'coeffs',
            'type': 'array',
            'variables': [{
                'name': 'a',
                'distribution': 'notvalid',
                'value': 0.5
            }]
        }]

        with self.assertRaises(ValidationError):
            validate(conf)
