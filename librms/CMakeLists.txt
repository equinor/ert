project(res-rms C)

add_library(rms src/rms_export.c
                src/rms_file.c
                src/rms_stats.c
                src/rms_tag.c
                src/rms_tagkey.c
                src/rms_type.c
                src/rms_util.c
)

target_link_libraries(rms PUBLIC ecl)
target_include_directories(rms
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)

set_target_properties(rms PROPERTIES
    VERSION ${RES_VERSION_MAJOR}.${RES_VERSION_MINOR}
    SOVERSION ${RES_VERSION_MAJOR}
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS rms
        EXPORT  res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (NOT BUILD_TESTS OR NOT STATOIL_TESTDATA_ROOT)
    return ()
endif ()

add_executable(rms_file_test tests/rms_file_test.c)
target_link_libraries(rms_file_test rms)

add_test(NAME rms_file_test
    COMMAND rms_file_test
            ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/rms_file_test/rms_file)
