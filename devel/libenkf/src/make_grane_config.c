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
  const char * grid_file = "/d/proj/bg/ior_fsenter2/grane/ressim/hstruct/2008a/e100/EnKF/sf02rg01/Refcase/GRANE.EGRID";
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
    
    /* North west */
    {
      const char * ministep_name = "NORTHW";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR01_G1");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR01_G1");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR01_G1");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR02_G12");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR02_G12");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR02_G12");
      
      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid ,  0,32,115,165,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }


    /* North centre */
    {
      const char * ministep_name = "NORTHC";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR03A_G8");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR03A_G8");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR03A_G8");

      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid , 33,43,115,165,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }

    /* North east 1 */
    {
      const char * ministep_name = "NORTHE1";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR06_G28");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR06_G28");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR06_G28");

      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid , 33,43,115,165,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }

    /* North east 2 */
    {
      const char * ministep_name = "NORTHE2";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR07_G02");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR07_G02");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR07_G02");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR08_G16");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR08_G16");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR08_G16");

      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid , 56, 101,115,165,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }


    /* Middle part */

    /* Middle west */

    {
      const char * ministep_name = "MIDDLEW";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR11E_G5");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR11E_G5");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR11E_G5");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR12_G19");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR12_G19");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR12_G19");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR11_G6");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR11_G6");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR11_G6");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR10_G18");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR10_G18");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR10_G18");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR09_G10");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR09_G10");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR09_G10");

      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid , 0, 45,89,114,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }


    /* Middle east */

    {
      const char * ministep_name = "MIDDLEE";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR13_G22");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR13_G22");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR13_G22");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR26_G26");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR26_G26");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR26_G26");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR14_G27");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR14_G27");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR14_G27");

      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid ,46, 101,89,114,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }


 /* Middle centre */

    {
      const char * ministep_name = "MIDDLEC";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR24_G17");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR24_G17");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR24_G17");

      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid ,0, 101,84,88,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }


    /*****************************************************************/
    /* The southern region. */

    /* South west */
    {
      const char * ministep_name = "SOUTHW";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR25_G21");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR25_G21");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR25_G21");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR27_G11");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR27_G11");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR27_G11");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR23_G9");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR23_G9");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR23_G9");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR22_G25");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR22_G25");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR22_G25");
      
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR29_G34");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR29_G34");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR29_G34");

      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid , 0,53,45,83,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }
  

  /* South east */
    {
      const char * ministep_name = "SOUTHE";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR18_G40");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR18_G40");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR18_G40");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR16_G15");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR16_G15");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR16_G15");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR17_G30");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR17_G30");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR17_G30");

      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR15_G35");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR15_G35");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR15_G35");
      
      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid , 54,101,45,83,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }
 

 /* South centre */
    {
      const char * ministep_name = "SOUTHC";
      /* Create the ministep */
      fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( CREATE_MINISTEP ) , ministep_name);
      
      /* Add observations */
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WWPR:PR20_G39");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WOPR:PR20_G39");
      fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name , "WGPR:PR20_G39");

      {
        ecl_box_type * ecl_box = ecl_box_alloc(ecl_grid , 0,101,0,44,0,19);
        /* Add the field data which is updated. */
        add_ecl_box(ministep_name , "PORO"     , ecl_box , stream );
        add_ecl_box(ministep_name , "PERMX"    , ecl_box , stream );
	add_ecl_box(ministep_name , "MULTPV"   , ecl_box , stream );
        add_ecl_box(ministep_name , "PRESSURE" , ecl_box , stream );
        
        ecl_box_free( ecl_box );
      }
      
      /* Attach the ministep to the updatestep */
      fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
    }
 
    /* Seismic update */
    
    /**
       {
       const char * ministep_name = "SEISMIC";
       
       
       fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( COPY_MINISTEP ) , "ALL_ACTIVE", ministep_name);
       
       fprintf(stream , "%-32s %s \n" , local_config_get_cmd_string( DEL_ALL_OBS ) , ministep_name);
       fprintf(stream , "%-32s %s %s \n" , local_config_get_cmd_string( ADD_OBS ) , ministep_name, "GEN_TIMESHIFT");
 
      
       fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
       }

    */


    /* Global update */
    /**
       {
       const char * ministep_name = "GLOBAL";
       
       
       fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( COPY_MINISTEP ) , "ALL_ACTIVE", ministep_name);
      
       fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( DEL_ALL_DATA_) , ministep_name);
       
       fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( DEL_OBS ) , ministep_name , "GEN_TIMESHIFT");
       fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( DEL_DATA ) , ministep_name , "TIMESHIFT");

       fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_DATA ) , ministep_name , "VARIOSHALE");
       fprintf(stream , "%-32s %s %s\n" , local_config_get_cmd_string( ADD_DATA ) , ministep_name , "VARIOSAND");
      
       fprintf(stream , "%-32s %s %s\n", local_config_get_cmd_string( ATTACH_MINISTEP ) , update_step , ministep_name);
       }
    
    */
 
  
    /******************************************************************/
    /* Set the updatestep as the default update step */
    fprintf(stream , "%-32s %s\n", local_config_get_cmd_string( INSTALL_DEFAULT_UPDATESTEP ) , update_step );
    
    
    fclose( stream );
  }
}
 
