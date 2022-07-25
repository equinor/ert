import logging
from copy import deepcopy


class _Feature:
    def __init__(self, default_enabled, msg=None):
        self.is_enabled = default_enabled
        self.msg = msg


class FeatureToggling:
    _conf_original = {
        "new-storage": _Feature(
            default_enabled=False,
            msg=(
                "The new storage solution is experimental! "
                "Thank you for testing our new features."
            ),
        ),
        "experiment-server": _Feature(
            default_enabled=False,
            msg=(
                "The experiment server is a step towards becoming Cloud Native. "
                "Thanks for testing."
            ),
        ),
    }

    _conf = deepcopy(_conf_original)

    @staticmethod
    def is_enabled(feature_name):
        return FeatureToggling._conf[feature_name].is_enabled

    @staticmethod
    def add_feature_toggling_args(parser):
        for feature_name in FeatureToggling._conf.keys():
            parser.add_argument(
                f"--{FeatureToggling._get_arg_name(feature_name)}",
                action="store_true",
                help=f"Toggle {feature_name} (Warning: This is experimental)",
                default=False,
            )

    @staticmethod
    def update_from_args(args):
        args_dict = vars(args)
        for feature_name in FeatureToggling._conf.keys():

            arg_name = FeatureToggling._get_arg_name(feature_name)
            feature_name_escaped = arg_name.replace("-", "_")

            if feature_name_escaped in args_dict and args_dict[feature_name_escaped]:
                current_state = FeatureToggling._conf[feature_name].is_enabled
                FeatureToggling._conf[feature_name].is_enabled = not current_state

            if (
                FeatureToggling._conf[feature_name].is_enabled
                and FeatureToggling._conf[feature_name].msg is not None
            ):
                logger = logging.getLogger()
                logger.warning(FeatureToggling._conf[feature_name].msg)

    @staticmethod
    def _get_arg_name(feature_name):
        default_state = FeatureToggling._conf[feature_name].is_enabled
        arg_default_state = "disable" if default_state else "enable"
        return f"{arg_default_state}-{feature_name}"

    @staticmethod
    def reset():
        FeatureToggling._conf = deepcopy(FeatureToggling._conf_original)


def feature_enabled(feature_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if FeatureToggling.is_enabled(feature_name):
                return func(*args, **kwargs)
            else:
                return None

        return wrapper

    return decorator
