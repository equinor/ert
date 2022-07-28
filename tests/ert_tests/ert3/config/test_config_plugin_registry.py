def test_no_plugins(plugin_registry):
    assert plugin_registry.get_factory("transformation", "copy")
    assert plugin_registry.get_factory("transformation", "serialization")
    assert plugin_registry.get_factory("transformation", "summary")
    assert plugin_registry.get_factory("transformation", "directory")

    assert plugin_registry.get_descriminator("transformation") == "type"

    generated_configs = [
        "ert.ert3.config._config_plugin_registry.FullCopyTransformationConfig",
        "ert.ert3.config._config_plugin_registry.FullSerializationTransformationConfig",
        "ert.ert3.config._config_plugin_registry.FullSummaryTransformationConfig",
        "ert.ert3.config._config_plugin_registry.FullDirectoryTransformationConfig",
    ]
    assert (
        str(plugin_registry.get_type("transformation"))
        == f"typing.Union[{', '.join(generated_configs)}]"
    )

    field = plugin_registry.get_field("transformation")
    assert field.discriminator == "type"
