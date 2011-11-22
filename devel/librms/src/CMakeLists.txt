set( source_files rms_file.c rms_util.c rms_tag.c rms_type.c rms_tagkey.c rms_stats.c rms_export.c)
set( header_files rms_file.h rms_util.h rms_tag.h rms_type.h rms_tagkey.h rms_stats.h rms_export.h)

add_library( rms_static  STATIC ${source_files} )
add_library( rms_shared  SHARED ${source_files} )

set_target_properties( rms_static PROPERTIES OUTPUT_NAME rms )
set_target_properties( rms_shared PROPERTIES OUTPUT_NAME rms )

target_link_libraries( rms_static ${util_static} ${geometry_static} )
target_link_libraries( rms_shared ${util_shared} ${geometry_shared})

if (OPENMP)
   target_link_libraries( rms_static gomp )
   target_link_libraries( rms_shared gomp )
endif()


#-----------------------------------------------------------------
if (RMS_INSTALL_PREFIX)
   install(TARGETS rms_static rms_shared DESTINATION ${INSTALL_ROOT}/${RMS_INSTALL_PREFIX}/lib)
   install(FILES ${header_files} DESTINATION ${INSTALL_ROOT}/${RMS_INSTALL_PREFIX}/include)
else()
   install(TARGETS rms_static rms_shared DESTINATION ${INSTALL_ROOT}/lib)
   install(FILES ${header_files} DESTINATION ${INSTALL_ROOT}/include)
endif()        


