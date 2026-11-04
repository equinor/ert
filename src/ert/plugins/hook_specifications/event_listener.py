from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from ert.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.run_models.event import StatusEvents


@no_type_check
@hook_specification
def on_status_event(event: StatusEvents) -> None:
    """Called on every status event emit"""
