set( source_files well_state.c well_conn.c well_info.c well_ts.c well_branch.c well_path.c )
set( header_files well_state.h well_const.h well_conn.h well_info.h well_ts.h well_branch.h well_path.h )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${libutil_build_path} )
include_directories( ${libutil_src_path} )
include_directories( ${libgeometry_src_path} )
include_directories( ${libecl_src_path} )


add_library( well_static  STATIC ${source_files} )
add_library( well_shared  SHARED ${source_files} )

set_target_properties( well_static PROPERTIES OUTPUT_NAME well )
set_target_properties( well_shared PROPERTIES OUTPUT_NAME well )

target_link_libraries( well_static ecl_static )
target_link_libraries( well_shared ecl_shared )


#-----------------------------------------------------------------
if (WELL_INSTALL_PREFIX)
   install(TARGETS well_static well_shared DESTINATION ${CMAKE_INSTALL_PREFIX}/${WELL_INSTALL_PREFIX}/lib)
   install(FILES ${header_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/${WELL_INSTALL_PREFIX}/include)
else()
   install(TARGETS well_static well_shared DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
   install(FILES ${header_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()        


