set(PYTHON_SOURCES
    __init__.py
    ecl_config.py
    ecl_run.py
    script.py
)

# We look for binary 'flow' using normal cmake operations, if we indeed find
# flow we can create a valid flow entry in the ecl_config.yml file. That way the
# installation should be usable with flow with zero additional configuration.

set(_build_path "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
set(_flow_config_file "${_build_path}/flow_config.yml")
set(_ecl100_config_file "${_build_path}/ecl100_config.yml")
set(_ecl300_config_file "${_build_path}/ecl300_config.yml")

find_package(Flow)
if (Flow_FOUND)
  find_package(MPI)
  if (MPIEXEC)
    configure_file( flow_config_mpi.yml.in ${_flow_config_file})
  else()
    configure_file( flow_config.yml.in ${_flow_config_file})
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
  configure_file( flow_config_mpi.yml.in ${_flow_config_file})
endif()


configure_file( ecl100_config.yml "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res/fm/ecl/ecl100_config.yml" COPYONLY)
configure_file( ecl300_config.yml "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res/fm/ecl/ecl300_config.yml" COPYONLY)

add_python_package("python.res.fm.ecl" ${PYTHON_INSTALL_PREFIX}/res/fm/ecl "${PYTHON_SOURCES}" True)

install_example(${_flow_config_file}   "${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
install_example(${_ecl100_config_file} "${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
install_example(${_ecl300_config_file} "${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
