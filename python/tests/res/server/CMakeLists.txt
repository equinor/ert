set(TEST_SOURCES
    __init__.py
    test_simulation_context.py
    test_rpc_service.py
    test_rpc_storage.py
)

add_python_package("python.tests.res.server" ${PYTHON_INSTALL_PREFIX}/tests/res/server "${TEST_SOURCES}" False)

addPythonTest(res.server.simulation_context tests.res.server.test_simulation_context.SimulationContextTest)
addPythonTest(res.server.rpc_service tests.res.server.test_rpc_service.RPCServiceTest)
addPythonTest(res.server.rpc_storage tests.res.server.test_rpc_storage.RPCStorageTest)
