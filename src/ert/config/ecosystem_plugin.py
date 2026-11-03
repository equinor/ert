from __future__ import annotations

from abc import abstractmethod

from pydantic import BaseModel

from .parsing import ConfigDict


class EcosystemPlugin(BaseModel):
    """Base class for plugins selected with the ``ECOSYSTEM <name>`` keyword.

    An ecosystem plugin is given the parsed user configuration (the config
    dict) before it is turned into an :class:`ErtConfig`, and may modify it
    to insert hooks towards the ecosystem.

    Options given as ``KEY:VALUE`` after the name are passed to
    :meth:`configure` as keyword arguments with lower-cased keys, so the
    plugin decides which options exist by declaring them as parameters.
    """

    name: str

    @abstractmethod
    def configure(self, config_dict: ConfigDict, **options: str) -> None:
        """Modify ``config_dict`` in place according to ``options``."""

    def on_gui_startup(self) -> None:
        """Called once when ``ert gui`` has read a configuration selecting this plugin.

        Runs on the Qt main thread before the main window is shown, so the
        plugin may open dialogs, for instance to verify that credentials
        needed by its hooks and forward model steps are available.
        """
