set( source_files well_state.c well_conn.c well_info.c well_ts.c)
set( header_files well_state.c well_const.h well_conn.h well_info.h well_ts.h )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${libutil_build_path} )
include_directories( ${libutil_src_path} )
include_directories( ${libgeometry_src_path} )
include_directories( ${libecl_src_path} )


add_library( well_static  STATIC ${source_files} )
add_library( well_shared  SHARED ${source_files} )

set_target_properties( well_static PROPERTIES OUTPUT_NAME well )
set_target_properties( well_shared PROPERTIES OUTPUT_NAME well )

target_link_libraries( well_static ${ecl_static} ${util_static} ${geometry_static} )
target_link_libraries( well_shared ${ecl_shared} ${util_shared} ${geometry_shared})


#-----------------------------------------------------------------
if (WELL_INSTALL_PREFIX)
   install(TARGETS well_static well_shared DESTINATION ${INSTALL_ROOT}/${WELL_INSTALL_PREFIX}/lib)
   install(FILES ${header_files} DESTINATION ${INSTALL_ROOT}/${WELL_INSTALL_PREFIX}/include)
else()
   install(TARGETS well_static well_shared DESTINATION ${INSTALL_ROOT}/lib)
   install(FILES ${header_files} DESTINATION ${INSTALL_ROOT}/include)
endif()        


