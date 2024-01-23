import logging
import os
from argparse import ArgumentParser
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
    from ert.namespace import Namespace


class _Feature:
    def __init__(
        self, default: Optional[bool], msg: Optional[str] = None, optional: bool = False
    ) -> None:
        self._value = default
        self.msg = msg
        self.optional = optional

    def validate_value(self, value: Union[bool, str, None]) -> Optional[bool]:
        if type(value) is bool or value is None:
            return value
        elif value.lower() in ["true", "1"]:
            return True
        elif value.lower() in ["false", "0"]:
            return False
        elif self.optional and value.lower() in ["default", ""]:
            return None
        else:
            raise ValueError(
                f"This option can only be set to {'True/1, False/0 or Default/<empty>' if self.optional else 'True/1 or False/0'}"
            )

    @property
    def value(self) -> Optional[bool]:
        return self._value

    @value.setter
    def value(self, value: Optional[bool]) -> None:
        self._value = self.validate_value(value)


class FeatureToggling:
    _conf_original: Dict[str, _Feature] = {
        "scheduler": _Feature(
            default=None,
            msg="Default value for use of Scheduler has been overridden\n"
            "This is experimental and may cause problems",
            optional=True,
        ),
    }

    _conf = deepcopy(_conf_original)

    @staticmethod
    def is_enabled(feature_name: str) -> bool:
        return FeatureToggling._conf[feature_name].value is True

    @staticmethod
    def value(feature_name: str) -> Optional[bool]:
        return FeatureToggling._conf[feature_name].value

    @staticmethod
    def add_feature_toggling_args(parser: ArgumentParser) -> None:
        for name, feature in FeatureToggling._conf.items():
            env_var_name = f"ERT_FEATURE_{name.replace('-', '_').upper()}"
            env_value: Union[bool, str, None] = None
            if env_var_name in os.environ:
                try:
                    feature.value = feature.validate_value(os.environ[env_var_name])
                except ValueError as e:
                    # TODO: this is a bit spammy. It will get called 6 times for each incorrect env var.
                    logging.getLogger().warning(
                        f"Failed to set {env_var_name} to '{os.environ[env_var_name]}'. {e}"
                    )

            if not feature.optional:
                parser.add_argument(
                    f"--{'disable' if feature.value else 'enable'}-{name}",
                    action="store_false" if feature.value else "store_true",
                    help=f"Toggle {name} (Warning: This is experimental)",
                    dest=f"feature-{name}",
                    default=env_value if env_value is not None else feature.value,
                )
            else:
                group = parser.add_mutually_exclusive_group()
                group.add_argument(
                    f"--enable-{name}",
                    action="store_true",
                    help=f"Enable {name}",
                    dest=f"feature-{name}",
                    default=feature.value,
                )
                group.add_argument(
                    f"--disable-{name}",
                    action="store_false",
                    help=f"Disable {name}",
                    dest=f"feature-{name}",
                    default=feature.value,
                )

    @staticmethod
    def update_from_args(args: "Namespace") -> None:
        pattern = "feature-"
        feature_args = [arg for arg in vars(args).items() if arg[0].startswith(pattern)]
        for name, value in feature_args:
            name = name[len(pattern) :]
            if name in FeatureToggling._conf:
                FeatureToggling._conf[name].value = value

        # Print warnings for enabled features.
        for name, feature in FeatureToggling._conf.items():
            if FeatureToggling.is_enabled(name) and feature.msg is not None:
                logging.getLogger().warning(
                    f"{feature.msg}\nValue is set to {feature.value}"
                )

    @staticmethod
    def reset() -> None:
        FeatureToggling._conf = deepcopy(FeatureToggling._conf_original)
