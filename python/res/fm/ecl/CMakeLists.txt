set(PYTHON_SOURCES
    __init__.py
    ecl_config.py
    ecl_run.py
)

# We look for binary 'flow' using normal cmake operations, if we indeed find
# flow we can create a valid flow entry in the ecl_config.yml file. That way the
# installation should be usable with flow with zero additional configuration.

set(_ecl_config_file "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res/fm/ecl/ecl_config.yml")
find_package(Flow)
if (Flow_FOUND)
  find_package(MPI)
  if (MPIEXEC)
    configure_file( ecl_config_mpi.yml.in ${_ecl_config_file})
  else()
    configure_file( ecl_config.yml.in ${_ecl_config_file})
  endif()
endif()


if (NOT Flow_FOUND)
  # flow has not been found we just make a ecl_config.yml file
  # with dummy paths to the location of flow and the mpiexec
  # executable, the resulting ecl_config.yml file must be
  # edited before it can be used.
  set(FLOW_EXECUTABLE "/example/path/to/flow")
  set(FLOW_VERSION "2018.10")
  set(MPIEXEC "/example/path/to/mpiexec")
  configure_file( ecl_config_mpi.yml.in ${_ecl_config_file})
endif()

add_python_package("python.res.fm.ecl" ${PYTHON_INSTALL_PREFIX}/res/fm/ecl "${PYTHON_SOURCES}" True)
install_example(${_ecl_config_file} "${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
