from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger()


if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ert.config.parsing.queue_system import QueueSystem
    from ert.namespace import Namespace


class FeatureScheduler:
    _DEFAULTS = {
        "LOCAL": True,
        "LSF": True,
        "SLURM": False,
        "TORQUE": True,
    }
    _value: Optional[bool] = None

    @classmethod
    def is_enabled(cls, queue_system: QueueSystem) -> bool:
        if queue_system.name == "TORQUE":
            return True
        if cls._value is not None:
            return cls._value
        return cls._DEFAULTS[queue_system.name]

    @classmethod
    def set_value(cls, args: Namespace) -> None:
        if ((value := cls._get_from_args(args)) is not None) or (
            (value := cls._get_from_env()) is not None
        ):
            cls._value = value
        else:
            cls._value = None

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> None:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--enable-scheduler",
            action="store_true",
            help="Enable new scheduler",
            dest="feature_scheduler",
            default=None,
        )
        group.add_argument(
            "--disable-scheduler",
            action="store_false",
            help="Disable new scheduler",
            dest="feature_scheduler",
            default=None,
        )

    @staticmethod
    def _get_from_env() -> Optional[bool]:
        if (value := os.environ.get("ERT_FEATURE_SCHEDULER")) is None:
            return None
        value = value.lower()
        if value in ("true", "1"):
            return True
        elif value in ("false", "0"):
            return False
        elif value in ("auto", "default", ""):
            return None
        raise ValueError(
            "This option can only be set to 'true'/'1', 'false'/'0' or 'auto'/'default'/''"
        )

    @staticmethod
    def _get_from_args(args: Namespace) -> Optional[bool]:
        if hasattr(args, "feature_scheduler"):
            return args.feature_scheduler
        return None
