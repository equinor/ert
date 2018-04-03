set(PYTHON_SOURCES
    __init__.py
    ertrpcclient.py
    ertrpcserver.py
    simulation_context.py
)

add_python_package("python.res.server" ${PYTHON_INSTALL_PREFIX}/res/server "${PYTHON_SOURCES}" True)

