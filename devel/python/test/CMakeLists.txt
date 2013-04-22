add_python_package("Python test" "${PYTHON_INSTALL_PREFIX}/test" "ecl_isosurf.py;enkf_test.py;ens_config_test.py;file_test.py;fortio_test.py;grav_test.py;grdecl_test.py;grid_test0.py;grid_test.py;import_test.py;job_test.py;kw_test.py;large_mem_test.py;latex_test.py;petrel_kw.py;poly_test.py;region_test.py;restart_test.py;rft_test.py;sched_test.py;stringlist_test.py;sum_test.py;test_all.py;test_fast.py;test_util.py;troll_test.py;util_test.py;ctest_run.py" OFF)

set(CTEST_PYTHON ${CMAKE_CURRENT_BINARY_DIR}/ctest_run.py)


add_test( python.ert.ecl.ecl_sum python ${CTEST_PYTHON} ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX} sum_test )
add_test( python.import_all      python ${CTEST_PYTHON} ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX} import_test )


set_property( TEST python.ert.ecl.ecl_sum  PROPERTY LABELS Python:Statoil )
set_property( TEST python.import_all       PROPERTY LABELS Python )
