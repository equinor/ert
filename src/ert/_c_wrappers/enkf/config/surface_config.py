from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class SurfaceConfig(BaseCClass):
    TYPE_NAME = "surface_config"

    _base_surface_path = ResPrototype(
        "char* surface_config_base_surface_path(surface_config)"
    )

    def __repr__(self):
        return f"SurfaceConfig() {self._ad_str()}"

    @property
    def base_surface_path(self) -> str:
        return self._base_surface_path()
