add_python_target( python_gui_pages_config ${PYTHON_INSTALL_PREFIX}/ert_gui/pages/config "__init__;analysis;configpages;eclipse;ensemble;observations;plot;queuesystem;simulation;systemenv")
add_subdirectory( jobs )
add_subdirectory( parameters )
add_subdirectory( simulations )
