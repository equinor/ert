add_definitions( -DINTERNAL_LINK )
set( source_files analysis_module.c enkf_linalg.c std_enkf.c sqrt_enkf.c cv_enkf.c bootstrap_enkf.c)
set( header_files analysis_module.h enkf_linalg.h analysis_table.h)

add_library( analysis_shared  SHARED ${source_files} )
add_library( analysis_static  SHARED ${source_files} )

set_target_properties( analysis_shared PROPERTIES OUTPUT_NAME analysis )
set_target_properties( analysis_static PROPERTIES OUTPUT_NAME analysis )

target_link_libraries( analysis_static ${util_static} )
target_link_libraries( analysis_shared ${util_shared} )

#remove_definitions( -DINTERNAL_LINK ) 
#add_library( std_enkf  MODULE std_enkf.c )
#add_library( sqrt_enkf MODULE sqrt_enkf.c )


#-----------------------------------------------------------------

if (ANALYSIS_INSTALL_PREFIX)
   install(TARGETS analysis_shared analysis_static DESTINATION ${INSTALL_ROOT}/${ANALYSIS_INSTALL_PREFIX}/lib)
   install(FILES ${header_files} DESTINATION ${INSTALL_ROOT}/${ANALYSIS_INSTALL_PREFIX}/include)
else()
   install(TARGETS analysis_shared analysis_static DESTINATION ${INSTALL_ROOT}/lib)
   install(FILES ${header_files} DESTINATION ${INSTALL_ROOT}/include)
endif()        


