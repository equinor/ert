#include <ecl_grid.h>
#include <local_config.h>
#include <util.h>




int main( int argc , char ** argv) {
  if (argc != 4) {
    fprintf(stderr,"Usage:\n\nbash%% local_config  GRID_FILE   NEW_CONFIG_FILE  OLD_CONFIG_FILE");
    exit(1);
  } else {
    ecl_grid_type * ecl_grid = ecl_grid_alloc( argv[1] );
    const char * src_file    = argv[2];
    const char * target_file = argv[3]; 

    local_config_type * local_config = local_config_alloc( 100 );
    local_config_add_config_file( local_config , src_file );
    local_config_reload( local_config , ecl_grid , NULL , NULL , NULL );
    
    local_config_fprintf( local_config , target_file );
    local_config_free( local_config );
  }
}
