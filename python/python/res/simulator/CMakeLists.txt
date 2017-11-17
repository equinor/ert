set(PYTHON_SOURCES
    __init__.py
    batch_simulator.py
    batch_simulator_context.py)

add_python_package("python.res.simulator" ${PYTHON_INSTALL_PREFIX}/res/simulator "${PYTHON_SOURCES}" True)
