import ert3


_modified_classes = {
    ("ert3.config", "StageIO"): ["transformation"],
    ("ert3.config", "EnsembleInput"): ["transformation"],
}
_all_affected_modules = ["ert3.config"]


def process_docstring_callback(app, what, name, obj, options, lines):
    # This sphinx directive will inject plugin information to all class
    # Specified in _modified_classes. Since the referenced configs are created
    # dynamically, they cannot be referenced. Hence, the type info is removed,
    # and we rather show a reference to 'Plugin configuration reference'
    # where we have generated documentation for each plugin category.

    if what != "module" and name not in _all_affected_modules:
        return

    for (module, modified_cls), categories in _modified_classes.items():
        plugin_registry = ert3.config.ConfigPluginRegistry()
        for category in categories:
            plugin_registry.register_category(
                category=category,
                base_config=ert3.config.plugins.TransformationConfigBase,
                optional=True,
            )
        plugin_manager = ert3.plugins.ErtPluginManager()
        plugin_manager.collect(registry=plugin_registry)

        if name != module:
            continue

        base_cls = getattr(obj, modified_cls)
        cls_transformed = ert3.config.create_plugged_model(
            model_name=modified_cls,
            categories=categories,
            plugin_registry=plugin_registry,
            model_base=base_cls,
            model_module=obj.__name__,
            docs=True,
        )
        for category in categories:
            setattr(
                cls_transformed,
                category,
                (
                    f"See section '{category}' under 'Plugin configuration reference' "
                    "for all valid configs"
                ),
            )

        setattr(obj, modified_cls, cls_transformed)


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring_callback)
