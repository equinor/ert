set(TEST_SOURCES
    __init__.py
   test_batch_sim.py)

add_python_package("python.tests.res.simulator" ${PYTHON_INSTALL_PREFIX}/tests/res/simulator "${TEST_SOURCES}" False)

addPythonTest(tests.res.simulator.test_batch_sim.BatchSimulatorTest)
