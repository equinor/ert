set( source_files config.c config_schema_item.c config_content_node.c conf.c conf_util.c conf_data.c)
set( header_files config.h config_schem_item.h config_content_node.h conf.h conf_data.h)

add_library( config ${LIBRARY_TYPE} ${source_files} )

target_link_libraries( config ert_util )

if (CONFIG_INSTALL_PREFIX)
   install(TARGETS config DESTINATION ${CONFIG_INSTALL_PREFIX}/lib)
   install(FILES ${header_files} DESTINATION ${CONFIG_INSTALL_PREFIX}/include)
else()
   install(TARGETS config DESTINATION lib)
   install(FILES ${header_files} DESTINATION include)
endif()
