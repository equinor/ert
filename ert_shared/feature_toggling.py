class FeatureToggling:
    __conf = {
        "new-storage": False,
    }

    @staticmethod
    def is_enabled(feature_name):
        return FeatureToggling.__conf[feature_name]

    @staticmethod
    def add_feature_toggling_args(parser):
        for feature_name in FeatureToggling.__conf.keys():
            parser.add_argument(
                "--{}".format(FeatureToggling._get_arg_name(feature_name)),
                action="store_true",
                help="Toggle {} (Warning: This is experiemental)".format(feature_name),
                default=False,
            )

    @staticmethod
    def update_from_args(args):
        args_dict = vars(args)
        for feature_name in FeatureToggling.__conf.keys():
            arg_name = FeatureToggling._get_arg_name(feature_name)
            feature_name_escaped = arg_name.replace("-", "_")

            if feature_name_escaped in args_dict:
                FeatureToggling.__conf[feature_name] = args_dict[feature_name_escaped]

    @staticmethod
    def _get_arg_name(feature_name):
        default_state = FeatureToggling.__conf[feature_name]
        arg_default_state = "disable" if default_state else "enable"
        return "{}-{}".format(arg_default_state, feature_name)
