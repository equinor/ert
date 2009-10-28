#include <IrapClassicMap.h>
#include <EclGrid.hpp>
#include <EclRegion.hpp>
#include <util.h>
#include <stdio.h>
#include <local_config.h>
#include <intVector.hpp>
#include <math.h>
#include <geom/space.h>



static void include_cell(const EclGrid * Grid , int global_index , const IrapClassicMap * Map , intVector * indexList) {
  const int num_points = 5;
  double x,y,z;
  Point2D p0;
  Point2D p1;
  Point2D p2;
  Vector2D v1;
  Vector2D v2;
  
  
  Grid->get_corner_xyz1(global_index , 0 , x , y , z);
  p0 = Point2D( x , y );
  Grid->get_corner_xyz1(global_index , 1 , x , y , z);
  p1 = Point2D( x , y );
  Grid->get_corner_xyz1(global_index , 2 , x , y , z);
  p2 = Point2D( x , y );
  
  v1 = Vector2D(p0 , p1);
  v2 = Vector2D(p0 , p2);
  {
    int i,j;
    Point2D p;
    for (j=0; j < num_points; j++)
      for (i =0; i < num_points; i++) {
        p = p0 + v1*(i + 0.5) * (1/num_points) + v2*(j + 0.5) * (1/ num_points);
        
        {
          int map_i , map_j, err;
          Map->getIndex( x , y , map_i , map_j , err);
          if (err == 0)
            indexList->append( Map->getGlobalIndex( map_i , map_j ));
        }
      }
  }
}


static void fprintf_command(FILE * stream , local_config_instruction_type cmd) {
  fprintf(stream , "%-32s " , local_config_get_cmd_string( cmd ));
}



static void add_eclipse_field( FILE * stream , const char * ministep , const char * ecl_key , const EclRegion * Region) {
  fprintf_command( stream , ADD_DATA );
  fprintf(stream , "%s %s \n", ministep , ecl_key);
  fprintf_command( stream , ACTIVE_LIST_ADD_MANY_DATA_INDEX );
  fprintf(stream , "%s %s %d\n" , ministep , ecl_key , Region->get_active_size());
  {
    int i;
    const int * active_list = Region->get_active_list();
    for (i=0; i < Region->get_active_size(); i++) {
      fprintf(stream , "%6d " , active_list[i]);
      if ((i % 10) == 9)
        fprintf(stream , "\n");
    }
  }
}



static void add_surface_data( FILE * stream , 
                              const char * ministep , 
                              const char * surface_key , 
                              const IrapClassicMap * Map , 
                              const EclGrid * Grid , 
                              const EclRegion * Region) {
  
  intVector   indexList         = intVector(0 , 0);      
  int         global_size       = Region->get_global_size();
  const int * global_index_list = Region->get_global_list();
  for (int i=0; i < global_size; i++) 
    include_cell(Grid , global_index_list[i] , Map , &indexList);
  
  indexList.select_unique();
  {
    fprintf_command( stream , ADD_DATA );
    fprintf(stream , "%s %s \n", ministep , surface_key);
    fprintf_command( stream , ACTIVE_LIST_ADD_MANY_DATA_INDEX );
    fprintf(stream , "%s %s %d\n" , ministep , surface_key , indexList.size());
    {
      int i;
      for (i=0; i < indexList.size(); i++) {
        fprintf(stream , "%6d " , indexList.iget(i));
        if ((i % 10) == 9)
          fprintf(stream , "\n");
      }
    }
  }
}




static void add_obs(FILE * stream , const char * ministep_name , const char * obs_key) {
  fprintf_command( stream , ADD_OBS );
  fprintf(stream , "%s %s\n",ministep_name , obs_key);
}



static void add_ministep(FILE * stream , const char * update_step , const char * ministep_name ) {
  fprintf_command( stream , CREATE_MINISTEP );
  fprintf(stream , "%s\n", ministep_name);
  fprintf_command( stream , ATTACH_MINISTEP );
  fprintf(stream , "%s %s\n", update_step , ministep_name);
}



/*****************************************************************/
/*
  To compile this program:

  1. Extract the python script below and store it in the current
     directory.  

  2. Run the the python script. This should produce a file
     'grane_local_config' which can be run to generate an input file
     for local analysis with ert/enkf.

  3. When using the program you should give the name of the resulting
     config file as the first and only argument on the command line.


Python start:
-----------------------------------------------------------------     
#!/usr/bin/python
import os
include_path_oddvar = "/private/joaho/EnKF/RMS-Oddvar/librms"
project_res_include = "/project/res/x86_64_RH_4/inc"
root         = "/private/joaho/EnKF/devel/EnKF"




include_path = ["%s/libutil/include" % root,
                "%s/libutil/cpp_src" % root,
                "%s/libecl/cpp_src"  % root,
                "%s/libecl/include"  % root,
                "%s/libenkf/include" % root,
                include_path_oddvar ,
                project_res_include]

lib_path = ["%s/libutil/lib" % root,
            "%s/libutil/cpp_src" % root,
            "%s/libecl/lib" % root,
            "%s/libecl/cpp_src" % root,
            include_path_oddvar]

lib_list    = ["-lrms" , "-lEcl" , "-lecl" , "-lutil" , "-lpt" , "-lz" , "-llapack"]
object_list = ["grane_local_config.o" , "active_list.o" ,   "local_config.o" ,   "local_ministep.o" ,   "local_updatestep.o"]
            

def make_string(l,p=""):
    s = ""
    for e in l:
        s += "%s%s  " % (p,e)
    return s


compile_cmd = "g++ -Wall -c grane_local_config.C %s" % make_string( include_path , "-I")
link_cmd    = "g++ -o grane_local_config %s %s %s" % (make_string(object_list) , make_string(lib_path , "-L") , make_string( lib_list ))

print "Compiling: %s" % compile_cmd
os.system( compile_cmd )

print "\nLinking: %s" % link_cmd
os.system( link_cmd )
-----------------------------------------------------------------     
Python end:
*/
  


int main (int argc , char ** argv ) {
  const char * command_file = argv[1];  /* The file to generate should be the first argument */
  const char * grid_file    = "/d/proj/bg/ior_fsenter2/grane/ressim/hstruct/2008a/e100/EnKF/sf02rg01/Refcase/GRANE.EGRID";
  const char * surface_file = "/d/proj/bg/ior_fsenter2/grane/ressim/hstruct/2008a/e100/EnKF/sf02rg01/Realizations/Cohiba/Surface_T/Init/topHeimdal_100.irap";
  FILE * stream = util_mkdir_fopen( command_file , "w");

  /* Load the grid */
  EclGrid   Grid    = EclGrid(grid_file);
  EclRegion Region  = EclRegion( &Grid , false );

  /* Load a surface */
  IrapClassicMap Map = IrapClassicMap();
  Map.read(surface_file);

  {
    const char * update_step = "UPDATESTEP";
    fprintf_command( stream , CREATE_UPDATESTEP );
    fprintf(stream , "%s\n", update_step);


    /* Region NORTHW */
    {
      const char * ministep_name = "NORTHW";
      add_ministep( stream , update_step , ministep_name );
      
      add_obs(stream , ministep_name , "WWPR:PR01_G1");
      add_obs(stream , ministep_name , "WOPR:PR01_G1");
      add_obs(stream , ministep_name , "WGPR:PR01_G1");
      
      add_obs(stream , ministep_name , "WWPR:PR02_G12");
      add_obs(stream , ministep_name , "WOPR:PR02_G12");
      add_obs(stream , ministep_name , "WGPR:PR02_G12");

      Region.select_from_ijkbox( 0,32,115,216,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "MULTPV"   , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);

      
      Region.select_from_ijkbox( 0,32,115,216, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE" , &Map , &Grid , &Region);
    }


    fprintf_command( stream , INSTALL_DEFAULT_UPDATESTEP );
    fprintf(stream , "%s\n", update_step);
    fclose(stream);
  }
}
