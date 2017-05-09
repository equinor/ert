set(PYTHON_SOURCES
    __init__.py
    oil_simulator.py
    perlin.py
    prime_generator.py
    shaped_perlin.py
)


add_python_package("python.res.test.synthesizer"  ${PYTHON_INSTALL_PREFIX}/res/test/synthesizer "${PYTHON_SOURCES}" True)
