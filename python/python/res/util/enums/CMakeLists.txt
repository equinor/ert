set(PYTHON_SOURCES
    __init__.py
    ui_return_status_enum.py
    message_level_enum.py
)

add_python_package("python.res.util.enums"  ${PYTHON_INSTALL_PREFIX}/res/util/enums "${PYTHON_SOURCES}" True)
