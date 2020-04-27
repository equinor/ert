import sys


class _Feature:

    def __init__(self, default_enabled, python3_only=False):
        self.is_python3_only = python3_only
        python_major_version = sys.version_info[0]
        self.is_enabled = default_enabled and not (python_major_version < 3 and python3_only)
        self.is_togglable = not (python_major_version < 3 and python3_only)

class FeatureToggling:
    __conf = {
        "new-storage": _Feature(default_enabled=False, python3_only=True),
    }

    @staticmethod
    def is_enabled(feature_name):
        return FeatureToggling.__conf[feature_name].is_enabled

    @staticmethod
    def add_feature_toggling_args(parser):
        for feature_name, feature in FeatureToggling.__conf.items():
            if not feature.is_togglable:
                continue

            parser.add_argument(
                "--{}".format(FeatureToggling._get_arg_name(feature_name)),
                action="store_true",
                help="Toggle {} (Warning: This is experiemental)".format(feature_name),
                default=False,
            )

    @staticmethod
    def update_from_args(args):
        args_dict = vars(args)
        for feature_name, feature in FeatureToggling.__conf.items():
            if not feature.is_togglable:
                continue

            arg_name = FeatureToggling._get_arg_name(feature_name)
            feature_name_escaped = arg_name.replace("-", "_")

            if feature_name_escaped in args_dict and args_dict[feature_name_escaped]:
                current_state = FeatureToggling.__conf[feature_name].is_enabled
                FeatureToggling.__conf[feature_name].is_enabled = not current_state

    @staticmethod
    def _get_arg_name(feature_name):
        default_state = FeatureToggling.__conf[feature_name].is_enabled
        arg_default_state = "disable" if default_state else "enable"
        return "{}-{}".format(arg_default_state, feature_name)
