#include <ecl_grid.h>
#include <ecl_box.h>
#include <util.h>
#include <local_config.h>


static void add_ecl_box( const char * ministep , const char * key , const ecl_box_type * ecl_box , FILE * stream) {
  /* Add the data node: */
  int active_size = ecl_box_get_active_size( ecl_box );
  fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_DATA ) , ministep , key);
  fprintf(stream , "%-32s %s %s %d\n" , local_config_get_cmd_string( ACTIVE_LIST_ADD_MANY_DATA_INDEX ) , ministep , key , active_size);
  {
    int i;
    const int * active_list = ecl_box_get_active_list( ecl_box );
    for (i=0; i < active_size; i++) {
      fprintf(stream ,"%6d ",active_list[i]);
      if ((i % 10) == 9)
        fprintf(stream , "\n");
    }
    fprintf(stream , "\n");
  }
}

static void usage( char * exe ) {
  fprintf(stderr,"Usage: \n\n");
  fprintf(stderr,"%s  NAME_OF_FILE\n\n" , exe);
  exit(1);
}


int main(int argc, char ** argv) {
  const char * command_file;
  const char * grid_file = "Gurbat/EXAMPLE_01_BASE.EGRID";
  ecl_grid_type * ecl_grid = ecl_grid_alloc( grid_file , true );
  
  if (argc < 2)
    usage(argv[0]);

  command_file = argv[1];
  {
    const char * update_step = "UPDATESTEP";
    FILE * stream = util_mkdir_fopen( command_file , "w");
    
    /* Create the updatestep: */
    fprintf(stream , "%-32s %s\n" , local_config_get_cmd_string( CREATE_UPDATESTEP ) , update_step);


    /*****************************************************************/
    /* The northern region. */
    {
      const char * ministep_name = "NORTH";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR01_G1");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR01_G2");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR01_G3");
      
      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid ,  1,4,1,5,1,10);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }

    /*****************************************************************/
    /* The southern region. */
    {
      const char * ministep_name = "SOUTH";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR01_G1");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR01_G2");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR01_G3");
      
      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid ,  5,8,6,10,5,12);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }
    
    /******************************************************************/
    /* Set the updatestep as the default update step */
    fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( INSTALL_DEFAULT_UPDATESTEP ) , update_step );
    
    
    fclose( stream );
  }
}
 
