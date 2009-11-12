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
          //printf("map_i:%d  map_j:%d   global:%d \n",map_i , map_j , Map->getGlobalIndex( map_i , map_j ));
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

      if ((Region->get_active_size() % 10) != 9)
        fprintf(stream , "\n");
    }
  }
}



static void add_surface_data( FILE * stream , 
                              const char * ministep , 
                              const char * surface_key , 
                              const IrapClassicMap * Map , 
                              const EclGrid        * Grid , 
                              const EclRegion      * Region) {
  
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
      if ((indexList.size() % 10) != 9)
        fprintf(stream , "\n");
    }
  }
  {
    char * filename = util_alloc_sprintf("Surfaces/%s.xy" , ministep);
    FILE * stream = util_mkdir_fopen( filename , "w");
    for (int l = 0; l < indexList.size(); l++) {
      int i, j;
      double x,y;
      Map->getIJ( indexList.iget(l) , i , j);
      Map->getXY( i , j , x , y);
      fprintf(stream , "%12.6f %12.6f \n", x , y);
    }
    fclose(stream);
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
  const char * grid_file    = "/d/proj/bg/enkf/jaskje/iior_hstruct_2009a_r004_local_struct/Refcase/RG09A-ST-PP-PRED-ROO4-1.EGRID";
  const char * surface_file = "/d/proj/bg/enkf/jaskje/iior_hstruct_2009a_r004_local_struct/Realizations/Cohiba/Surface_B/Init/baseHeimdal_0.irap";
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

    /* North west */
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
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);

      
      Region.select_from_ijkbox( 0,32,115,216, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }
    
    /* North centre */
    {
       const char * ministep_name = "NORTHC";
       add_ministep( stream , update_step , ministep_name );
       add_obs(stream , ministep_name , "WWPR:PR03A_G8");
       add_obs(stream , ministep_name , "WOPR:PR03A_G8");
       add_obs(stream , ministep_name , "WGPR:PR03A_G8");

       Region.select_from_ijkbox(33,43,115,216,0,19 );

       add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
       add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
       add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
       add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
       add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
       add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
       add_eclipse_field( stream , ministep_name , "RS" , &Region);

       Region.select_from_ijkbox(33,43,115,216, 19 , 19 );
       add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
       add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
       
    }
    
    /* North east 1 */
    {
       const char * ministep_name = "NORTHE1";
       add_ministep( stream , update_step , ministep_name );
       add_obs(stream , ministep_name , "WWPR:PR06_G28");
       add_obs(stream , ministep_name , "WOPR:PR06_G28");
       add_obs(stream , ministep_name , "WGPR:PR06_G28");
       
       Region.select_from_ijkbox(44,55,115,216,0,19 );

       add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
       add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
       add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
       add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
       add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
       add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
       add_eclipse_field( stream , ministep_name , "RS" , &Region);

       Region.select_from_ijkbox(44,55,115,216, 19 , 19 );
       add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
       add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    /* North east 2 */
    {
      const char * ministep_name = "NORTHE2";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR07_G2");
      add_obs(stream , ministep_name , "WOPR:PR07_G2");
      add_obs(stream , ministep_name , "WGPR:PR07_G2"); 
      
      add_obs(stream , ministep_name , "WWPR:PR08_G16");
      add_obs(stream , ministep_name , "WOPR:PR08_G16");
      add_obs(stream , ministep_name , "WGPR:PR08_G16"); 
      
      Region.select_from_ijkbox(56,101,115,216,0,19 );
      
      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(56,101,115,216, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
      
    }

    /* Middle part */
    /* Middle west */
    {
      const char * ministep_name = "MIDDLEW";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR11E_G5");
      add_obs(stream , ministep_name , "WOPR:PR11E_G5");
      add_obs(stream , ministep_name , "WGPR:PR11E_G5");
      
      add_obs(stream , ministep_name , "WWPR:PR12_G19");
      add_obs(stream , ministep_name , "WOPR:PR12_G19");
      add_obs(stream , ministep_name , "WGPR:PR12_G19");

      add_obs(stream , ministep_name , "WWPR:PR11_G6");
      add_obs(stream , ministep_name , "WOPR:PR11_G6");
      add_obs(stream , ministep_name , "WGPR:PR11_G6");
 
      add_obs(stream , ministep_name , "WWPR:PR10_G18");
      add_obs(stream , ministep_name , "WOPR:PR10_G18");
      add_obs(stream , ministep_name , "WGPR:PR10_G18");

      add_obs(stream , ministep_name , "WWPR:PR09_G10");
      add_obs(stream , ministep_name , "WOPR:PR09_G10");
      add_obs(stream , ministep_name , "WGPR:PR09_G10");

      Region.select_from_ijkbox(0,47,89,114,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(0,47,89,114, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    /* Middle east */
    {
      const char * ministep_name = "MIDDLEE1";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR13_G22");
      add_obs(stream , ministep_name , "WOPR:PR13_G22");
      add_obs(stream , ministep_name , "WGPR:PR13_G22");
      
      Region.select_from_ijkbox(46,54,89,114,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(46,54,89,114, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }
    
    {
      const char * ministep_name = "MIDDLEE2";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR26_G26");
      add_obs(stream , ministep_name , "WOPR:PR26_G26");
      add_obs(stream , ministep_name , "WGPR:PR26_G26");
     
      add_obs(stream , ministep_name , "WWPR:PR14_G27");
      add_obs(stream , ministep_name , "WOPR:PR14_G27");
      add_obs(stream , ministep_name , "WGPR:PR14_G27");
 
      Region.select_from_ijkbox(50,101,89,114,0,19 );
      
      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(50,101,89,114, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    /* Middle centre */
    {
      const char * ministep_name = "MIDDLEC";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR24_G17");
      add_obs(stream , ministep_name , "WOPR:PR24_G17");
      add_obs(stream , ministep_name , "WGPR:PR24_G17");

      Region.select_from_ijkbox(0,101,84,88,0,19 );
      
      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(0,101,84,88, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    /* The southern region. */
    /* South west */
    {
      const char * ministep_name = "SOUTHW1";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR25_G21");
      add_obs(stream , ministep_name , "WOPR:PR25_G21");
      add_obs(stream , ministep_name , "WGPR:PR25_G21");

      Region.select_from_ijkbox(0,101,78,84,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(0,101,78,84, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    {
      const char * ministep_name = "SOUTHW2";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR27_G11");
      add_obs(stream , ministep_name , "WOPR:PR27_G11");
      add_obs(stream , ministep_name , "WGPR:PR27_G11");

      Region.select_from_ijkbox(0,44,67,80,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(0,44,67,80, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    {
      const char * ministep_name = "SOUTHW3";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR23_G9");
      add_obs(stream , ministep_name , "WOPR:PR23_G9");
      add_obs(stream , ministep_name , "WGPR:PR23_G9");
      
      add_obs(stream , ministep_name , "WWPR:PR22_G25");
      add_obs(stream , ministep_name , "WOPR:PR22_G25");
      add_obs(stream , ministep_name , "WGPR:PR22_G25");

      add_obs(stream , ministep_name , "WWPR:PR29_G34");
      add_obs(stream , ministep_name , "WOPR:PR29_G34");
      add_obs(stream , ministep_name , "WGPR:PR29_G34");

      Region.select_from_ijkbox(39,53,45,80,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(39,53,45,80, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }

    /* South east */
    
    {
      const char * ministep_name = "SOUTHE1";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR18_G40");
      add_obs(stream , ministep_name , "WOPR:PR18_G40");
      add_obs(stream , ministep_name , "WGPR:PR18_G40");
      
      add_obs(stream , ministep_name , "WWPR:PR17_G30");
      add_obs(stream , ministep_name , "WOPR:PR17_G30");
      add_obs(stream , ministep_name , "WGPR:PR17_G30");
      
      Region.select_from_ijkbox(54,68,45,83,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(54,68,45,83, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }

    {
      const char * ministep_name = "SOUTHE2";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR16_G15");
      add_obs(stream , ministep_name , "WOPR:PR16_G15");
      add_obs(stream , ministep_name , "WGPR:PR16_G15");
      
      add_obs(stream , ministep_name , "WWPR:PR15_G35");
      add_obs(stream , ministep_name , "WOPR:PR15_G35");
      add_obs(stream , ministep_name , "WGPR:PR15_G35");
      
      Region.select_from_ijkbox(69,101,45,83,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(69,101,45,83, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
      
    }

    /* South centre */
    {
      const char * ministep_name = "SOUTHC";
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR20_G39");
      add_obs(stream , ministep_name , "WOPR:PR20_G39");
      add_obs(stream , ministep_name , "WGPR:PR20_G39");

      Region.select_from_ijkbox(0,101,0,44,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);
      
      Region.select_from_ijkbox(0,101,0,44, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }



    fprintf_command( stream , INSTALL_DEFAULT_UPDATESTEP );
    fprintf(stream , "%s\n", update_step);
    fclose(stream);
  }
}
