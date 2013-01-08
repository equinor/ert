set( source_files config.c config_error.c config_schema_item.c config_content_item.c config_content_node.c config_root_path.c config_path_elm.c conf.c conf_util.c conf_data.c)
set( header_files config.h config_error.h config_schem_item.h config_content_item.h config_content_node.h config_root_path.h config_path_elm.h conf.h conf_data.h)

add_library( config ${LIBRARY_TYPE} ${source_files} )

target_link_libraries( config ert_util )

if (CONFIG_INSTALL_PREFIX)
   install(TARGETS config DESTINATION ${CONFIG_INSTALL_PREFIX}/lib)
   install(FILES ${header_files} DESTINATION ${CONFIG_INSTALL_PREFIX}/include)
else()
   install(TARGETS config DESTINATION lib)
   install(FILES ${header_files} DESTINATION include)
endif()
