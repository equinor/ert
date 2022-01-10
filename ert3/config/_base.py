from functools import partial

from pydantic import BaseModel
from pydantic.main import ModelMetaclass

from ert3.config import PluginConfigManager


class ErtPluginField:
    pass


class ErtPluginMetaClass(ModelMetaclass):
    def __new__(cls, name, bases, attrs, **kwargs):
        if "plugin_manager" in kwargs:
            plugin_manager: PluginConfigManager = kwargs["plugin_manager"]

            def getter_template(self, field):
                config_instance = getattr(self, field)
                descriminator_value = getattr(
                    config_instance, plugin_manager.get_descriminator(category=field)
                )
                return plugin_manager.get_factory(
                    category=field, name=descriminator_value
                )(config_instance)

            plugin_fields = []
            if "__annotations__" in attrs:
                plugin_fields = [
                    k
                    for k, v in attrs["__annotations__"].items()
                    if v == ErtPluginField
                ]

            for field in plugin_fields:
                partial_func = partial(getter_template, field=field)
                attrs[f"get_{field}_instance"] = lambda self: partial_func(self)
                attrs[field] = plugin_manager.get_field(field)
                attrs["__annotations__"][field] = plugin_manager.get_type(field)

        return super().__new__(cls, name, bases, attrs)


class ErtBaseModel(BaseModel, metaclass=ErtPluginMetaClass):
    pass
