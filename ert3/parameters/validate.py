import configsuite
import yaml
from configsuite import MetaKeys as MK
from configsuite import types


class ValidationError(Exception):
    def __init__(self, errors):
        self.errors = errors


_valid_variable_props = {
    'uniform': ['min', 'max'],
    'const': ['value']
}


@configsuite.validator_msg("Is type a valid variable")
def _is_valid_variable(content):
    if content['distribution'] == 'uniform':
        return 'min' in content and 'max' in content and 'value' not in content
    elif content['distribution'] == 'const':
        return 'min' not in content and 'max' not in content and 'value' in content
    return False


schema_variables = {
    MK.Type: types.NamedDict,
    MK.Content: {
        "name": {MK.Type: types.String},
        "distribution": {
            MK.Type: types.String,
        },
        "min": {MK.Type: types.Number, MK.Required: False},
        "max": {MK.Type: types.Number, MK.Required: False},
        "value": {MK.Type: types.Number, MK.Required: False}
    },
    MK.ElementValidators: [_is_valid_variable]
}

_schema = {
    MK.Type: types.List,
    MK.Content: {
        MK.Item: {
            MK.Type: types.NamedDict,
            MK.Content: {
                "name": {
                    MK.Type: types.String
                },
                "type": {
                    MK.Type: types.String,
                },
                "variables": {
                    MK.Type: types.List,
                    MK.Content: {
                        MK.Item: schema_variables
                    }
                }
            }
        }
    }
}


def schema():
    return _schema


def validate(conf):
    suite = configsuite.ConfigSuite(conf, schema())
    if not suite.valid:
        raise ValidationError(suite.errors)
    return suite.snapshot


def validate_yaml(path):
    with open(path, 'r') as handle:
        conf = yaml.safe_load(handle)

    return validate(conf)
