project(ert-ert C)

if (NOT BUILD_SHARED_LIBS)
    message(WARNING "Building python - forcing shared libs.")
    set(BUILD_SHARED_LIBS ON)
endif ()

add_library(ert src/ert_version.c)

target_include_directories(ert
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)

target_link_libraries(ert PUBLIC ecl)
target_include_directories(ert
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)


target_compile_definitions(ert PRIVATE
            -DGIT_COMMIT=${GIT_COMMIT}
            -DGIT_COMMIT_SHORT=${GIT_COMMIT_SHORT}
            -DERT_VERSION_MAJOR=${ERT_VERSION_MAJOR}
            -DERT_VERSION_MINOR=${ERT_VERSION_MINOR}
            -DERT_VERSION_MICRO=${ERT_VERSION_MICRO}
            -DCOMPILE_TIME_STAMP="${ERT_BUILD_TIME}")


set_target_properties(ert PROPERTIES
    VERSION ${ERT_VERSION_MAJOR}.${ERT_VERSION_MINOR}
    SOVERSION ${ERT_VERSION_MAJOR})

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ert
        EXPORT  ert-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


add_subdirectory( applications/ert_tui )
