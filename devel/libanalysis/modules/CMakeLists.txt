set( RML_SOURCE_FILES
     rml_enkf.c
     rml_enkf_common.c )

set( RMLI_SOURCE_FILES
     rml_enkf_imodel.c
     rml_enkf_common.c )


#set( source_files rml_enkf.c rml_enkf_common.c rml_enkf_imodel.c )
set( header_files analysis_module.h enkf_linalg.h analysis_table.h std_enkf.h rml_enkf_common.h)
add_library( rml_enkf  SHARED ${RML_SOURCE_FILES} )
add_library( rmli_enkf  SHARED ${RMLI_SOURCE_FILES} )

set_target_properties( rml_enkf PROPERTIES COMPILE_DEFINITIONS INTERNAL_LINK)
set_target_properties( rml_enkf PROPERTIES VERSION 1.0 SOVERSION 1.0 )

set_target_properties( rmli_enkf PROPERTIES COMPILE_DEFINITIONS INTERNAL_LINK)
set_target_properties( rmli_enkf PROPERTIES VERSION 1.0 SOVERSION 1.0 )

target_link_libraries( rml_enkf analysis )
target_link_libraries( rml_enkf dl )
target_link_libraries( rmli_enkf analysis )
target_link_libraries( rmli_enkf dl )

if (USE_RUNPATH)
   add_runpath( rml_enkf )
endif()