project(res-sched C)

add_library(sched   src/group_history.c
                    src/group_index.c
                    src/gruptree.c
                    src/history.c
                    src/sched_blob.c
                    src/sched_file.c
                    src/sched_history.c
                    src/sched_kw.c
                    src/sched_kw_compdat.c
                    src/sched_kw_dates.c
                    src/sched_kw_gruptree.c
                    src/sched_kw_include.c
                    src/sched_kw_tstep.c
                    src/sched_kw_untyped.c
                    src/sched_kw_wconhist.c
                    src/sched_kw_wconinj.c
                    src/sched_kw_wconinje.c
                    src/sched_kw_wconinjh.c
                    src/sched_kw_wconprod.c
                    src/sched_kw_welspecs.c
                    src/sched_time.c
                    src/sched_types.c
                    src/sched_util.c
                    src/well_history.c
                    src/well_index.c
)

target_link_libraries(sched PUBLIC ecl)
target_include_directories(sched
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)

set_target_properties(sched PROPERTIES
    VERSION ${ERT_VERSION_MAJOR}.${ERT_VERSION_MINOR}
    SOVERSION ${ERT_VERSION_MAJOR}
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS sched
        EXPORT  res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (NOT BUILD_TESTS OR NOT STATOIL_TESTDATA_ROOT)
    return ()
endif ()

add_executable(sched_history_summary tests/sched_history_summary.c)
target_link_libraries(sched_history_summary sched)
add_test(NAME sched_history_summary1
         COMMAND sched_history_summary
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE)
add_test(NAME sched_history_summary2
         COMMAND sched_history_summary
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Snorre/SNORRE)
