set(TEST_SOURCES
    __init__.py
   test_batch_sim.py
    test_simulation_context.py)

add_python_package("python.tests.res.simulator" ${PYTHON_INSTALL_PREFIX}/tests/res/simulator "${TEST_SOURCES}" False)

python_config_test(tests.res.simulator.test_simulation_context.SimulationContextTest)
python_config_test(tests.res.simulator.test_batch_sim.BatchSimulatorTest)

