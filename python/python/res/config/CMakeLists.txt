set(PYTHON_SOURCES
    __init__.py
    schema_item.py
    config_error.py
    config_parser.py
    config_content.py
    config_settings.py
    content_type_enum.py
    unrecognized_enum.py
)

add_python_package("python.res.config"  ${PYTHON_INSTALL_PREFIX}/res/config "${PYTHON_SOURCES}" True)

