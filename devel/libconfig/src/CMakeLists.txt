set( source_files config.c config_schema_item.c conf.c conf_util.c conf_data.c)
set( header_files config.h config_schem_item.h conf.h conf_data.h)

add_library( config ${LIBRARY_TYPE} ${source_files} )

target_link_libraries( config ert_util )
