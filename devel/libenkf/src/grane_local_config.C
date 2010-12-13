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
        p = p0 + v1*(i + 0.5) * (1.0/num_points) + v2*(j + 0.5) * (1.0/ num_points);
        
        {
          int map_i , map_j, err;
          Map->getIndex( p.x() , p.y() , map_i , map_j , err);
          //printf("map_i:%d  map_j:%d   global:%d \n",map_i , map_j , Map->getGlobalIndex( map_i , map_j ));
          //printf("p.x() %g p.y() %g \n",p.x(),p.y());
          if (err == 0)
            indexList->append( Map->getGlobalIndex( map_i , map_j ));
          //printf("map %d \n", Map->getGlobalIndex( map_i , map_j ));
        }
      }
  }
  //  exit(1);
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
      fprintf(stream , "%12.6f %12.6f %12.6f  \n", x , y, 1640);
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


static void copy_ministep(FILE * stream, const char * ministep_copy, const char * ministep_name) {
  fprintf_command(stream, ALLOC_MINISTEP_COPY);
  fprintf(stream, "%s %s\n", ministep_copy , ministep_name); 
}

static void del_all_data(FILE * stream,  const char * ministep_name) {
  fprintf_command(stream, DEL_ALL_DATA);
  fprintf(stream , "%s\n", ministep_name);
}

static void del_all_obs(FILE * stream,  const char * ministep_name) {
  fprintf_command(stream, DEL_ALL_OBS);
  fprintf(stream , "%s\n", ministep_name);
}


static void add_data(FILE * stream,  const char * ministep_name, const char * data_key) {
  fprintf_command(stream, ADD_DATA);
  fprintf(stream , "%s %s \n", ministep_name, data_key);
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
                "%s/libjob_queue/src" % root , 
                "%s/libecl/cpp_src"  % root,
                "%s/libecl/include"  % root,
                "%s/libenkf/include" % root,
                include_path_oddvar ,
                project_res_include]

lib_path = ["%s/libutil/lib"     % root,
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
  const char * grid_file    = "/d/proj/bg/grane/IIOR/hstruct/2009b/users/ooy/r0020/enkf/eclipse/include/grid/PREDICTOR.EGRID";
  const char * surface_file = "/d/proj/bg/grane/IIOR/hstruct/2009b/users/ooy/r0020/enkf/realizations/cohiba/surface_topLowerHeimdal/Init/topLowerHeimdal_0.irap";

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

    
    /* Test region */
    /*
    {

      const char * ministep_name = "TEST_62_62_52_72";
      
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      
      add_obs(stream , ministep_name , "WWPT:PR01_G1");
      add_obs(stream , ministep_name , "WOPT:PR01_G1");
      add_obs(stream , ministep_name , "WGPT:PR01_G1");
      
      add_obs(stream , ministep_name , "WWPT:PR02_G12");
      add_obs(stream , ministep_name , "WOPT:PR02_G12");
      add_obs(stream , ministep_name , "WGPT:PR02_G12");

      Region.select_from_ijkbox( 62,62,52,72,0,19 );

      add_eclipse_field( stream , ministep_name , "PORO"     , &Region);
      add_eclipse_field( stream , ministep_name , "PERMX"    , &Region);
      add_eclipse_field( stream , ministep_name , "PERMZ"    , &Region);
      add_eclipse_field( stream , ministep_name , "PRESSURE" , &Region);
      add_eclipse_field( stream , ministep_name , "SWAT" , &Region);
      add_eclipse_field( stream , ministep_name , "SGAS" , &Region);
      add_eclipse_field( stream , ministep_name , "RS" , &Region);

      
      Region.select_from_ijkbox( 62,62,52,72, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }
    */


     /* 4D region 1-48 1-166 */
     /*
    {
       const char * ministep_name = "TIMESHIFTREGION";
       Region.Clear();
       add_ministep( stream , update_step , ministep_name );
       add_obs(stream , ministep_name , "GEN_TIMESHIFT");


       Region.select_from_ijkbox(0,101,50,130, 19 , 19 );
       add_surface_data( stream , ministep_name , "VELOCITY_T" , &Map , &Grid , &Region);
       add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }
     */

  /* North west */
    /* Region NORTHW */
    {
      const char * ministep_name = "NORTHW";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      
      add_obs(stream , ministep_name , "WWPR:PR01_G1");
      add_obs(stream , ministep_name , "WOPR:PR01_G1");
      add_obs(stream , ministep_name , "WOPT:PR01_G1");
      
      add_obs(stream , ministep_name , "WWPR:PR02_G12");
      add_obs(stream , ministep_name , "WOPR:PR02_G12");
      add_obs(stream , ministep_name , "WOPT:PR02_G12");

      add_obs(stream , ministep_name , "WWPR:PR34_G13");
      add_obs(stream , ministep_name , "WOPR:PR34_G13");
      add_obs(stream , ministep_name , "WOPT:PR34_G13");

      
      Region.select_from_ijkbox( 0,34,113,215, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }

   /* North centre */
    {
       const char * ministep_name = "NORTHC";
       Region.Clear();
       add_ministep( stream , update_step , ministep_name );
       add_obs(stream , ministep_name , "WWPR:PR03A_G8");
       add_obs(stream , ministep_name , "WOPR:PR03A_G8");
       add_obs(stream , ministep_name , "WOPT:PR03A_G8");

       Region.select_from_ijkbox(31,45,113,215, 19 , 19 );
       add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
       add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
       
    }
    
    /* North east 1 */
    {
       const char * ministep_name = "NORTHE1";
       Region.Clear();
       add_ministep( stream , update_step , ministep_name );
       add_obs(stream , ministep_name , "WWPR:PR06_G28");
       add_obs(stream , ministep_name , "WOPR:PR06_G28");
       add_obs(stream , ministep_name , "WOPT:PR06_G28");

       Region.select_from_ijkbox(42,57,113,215, 19 , 19 );
       add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
       add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    /* North east 2 */
    {
      const char * ministep_name = "NORTHE2";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR07_G2");
      add_obs(stream , ministep_name , "WOPR:PR07_G2");
      add_obs(stream , ministep_name , "WOPT:PR07_G2"); 
      
      add_obs(stream , ministep_name , "WWPR:PR08_G16");
      add_obs(stream , ministep_name , "WOPR:PR08_G16");
      add_obs(stream , ministep_name , "WOPT:PR08_G16"); 

      add_obs(stream , ministep_name , "WWPR:PR33_G3");
      add_obs(stream , ministep_name , "WOPR:PR33_G3");
      add_obs(stream , ministep_name , "WOPT:PR33_G3"); 
            
      Region.select_from_ijkbox(54,101,113,215, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
      
    }

    /* Middle part */
    /* Middle west */
    {
      const char * ministep_name = "MIDDLEW";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR11E_G5");
      add_obs(stream , ministep_name , "WOPR:PR11E_G5");
      add_obs(stream , ministep_name , "WOPT:PR11E_G5");
      
      add_obs(stream , ministep_name , "WWPR:PR12_G19");
      add_obs(stream , ministep_name , "WOPR:PR12_G19");
      add_obs(stream , ministep_name , "WOPT:PR12_G19");

      add_obs(stream , ministep_name , "WWPR:PR11_G6");
      add_obs(stream , ministep_name , "WOPR:PR11_G6");
      add_obs(stream , ministep_name , "WOPT:PR11_G6");
 
      add_obs(stream , ministep_name , "WWPR:PR10_G18");
      add_obs(stream , ministep_name , "WOPR:PR10_G18");
      add_obs(stream , ministep_name , "WOPT:PR10_G18");

      add_obs(stream , ministep_name , "WWPR:PR09_G10");
      add_obs(stream , ministep_name , "WOPR:PR09_G10");
      add_obs(stream , ministep_name , "WOPT:PR09_G10");

       
      Region.select_from_ijkbox(0,49,87,116, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    /* Middle east */
    {
      const char * ministep_name = "MIDDLEE1";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR13_G22");
      add_obs(stream , ministep_name , "WOPR:PR13_G22");
      add_obs(stream , ministep_name , "WOPT:PR13_G22");
      

      Region.select_from_ijkbox(44,56,87,116, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }
    
    {
      const char * ministep_name = "MIDDLEE2";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR26_G26");
      add_obs(stream , ministep_name , "WOPR:PR26_G26");
      add_obs(stream , ministep_name , "WOPT:PR26_G26");
     
      add_obs(stream , ministep_name , "WWPR:PR14_G27");
      add_obs(stream , ministep_name , "WOPR:PR14_G27");
      add_obs(stream , ministep_name , "WOPT:PR14_G27");

 
      Region.select_from_ijkbox(48,101,87,116, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    /* Middle centre */
    {
      const char * ministep_name = "MIDDLEC";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR24_G17");
      add_obs(stream , ministep_name , "WOPR:PR24_G17");
      add_obs(stream , ministep_name , "WOPT:PR24_G17");
      

      Region.select_from_ijkbox(0,100,82,90, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    /* The southern region. */
    /* South west */
    {
      const char * ministep_name = "SOUTHW1";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR25_G21");
      add_obs(stream , ministep_name , "WOPR:PR25_G21");
      add_obs(stream , ministep_name , "WOPT:PR25_G21");


      Region.select_from_ijkbox(0,100,76,86, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    {
      const char * ministep_name = "SOUTHW2";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR27_G11");
      add_obs(stream , ministep_name , "WOPR:PR27_G11");
      add_obs(stream , ministep_name , "WOPT:PR27_G11");

      
      Region.select_from_ijkbox(0,46,65,82, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);

    }

    {
      const char * ministep_name = "SOUTHW3";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR23_G9");
      add_obs(stream , ministep_name , "WOPR:PR23_G9");
      add_obs(stream , ministep_name , "WOPT:PR23_G9");
      
      add_obs(stream , ministep_name , "WWPR:PR22_G25");
      add_obs(stream , ministep_name , "WOPR:PR22_G25");
      add_obs(stream , ministep_name , "WOPT:PR22_G25");

      add_obs(stream , ministep_name , "WWPR:PR29_G34");
      add_obs(stream , ministep_name , "WOPR:PR29_G34");
      add_obs(stream , ministep_name , "WOPT:PR29_G34");
      
      Region.select_from_ijkbox(37,55,43,82, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }

    /* South east */
    
    {
      const char * ministep_name = "SOUTHE1";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR18_G40");
      add_obs(stream , ministep_name , "WOPR:PR18_G40");
      add_obs(stream , ministep_name , "WOPT:PR18_G40");
      
      add_obs(stream , ministep_name , "WWPR:PR17_G30");
      add_obs(stream , ministep_name , "WOPR:PR17_G30");
      add_obs(stream , ministep_name , "WOPT:PR17_G30");


      Region.select_from_ijkbox(52,70,43,85, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }

    {
      const char * ministep_name = "SOUTHE2";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR16_G15");
      add_obs(stream , ministep_name , "WOPR:PR16_G15");
      add_obs(stream , ministep_name , "WOPT:PR16_G15");
      
      add_obs(stream , ministep_name , "WWPR:PR15_G35");
      add_obs(stream , ministep_name , "WOPR:PR15_G35");
      add_obs(stream , ministep_name , "WOPT:PR15_G35");
      

      Region.select_from_ijkbox(67,100,43,85, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
      
    }

    /* South centre */
    {
      const char * ministep_name = "SOUTHC";
      Region.Clear();
      add_ministep( stream , update_step , ministep_name );
      add_obs(stream , ministep_name , "WWPR:PR20_G39");
      add_obs(stream , ministep_name , "WOPR:PR20_G39");
      add_obs(stream , ministep_name , "WOPT:PR20_G39");
      

      Region.select_from_ijkbox(1,100,1,44, 19 , 19 );
      add_surface_data( stream , ministep_name , "SURFACE_T" , &Map , &Grid , &Region);
      add_surface_data( stream , ministep_name , "SURFACE_B" , &Map , &Grid , &Region);
    }

    /* Global UPDATE */


    {
      const char * ministep_name = "GLOBAL_FIELD_OBS_PETRO";
      Region.Clear();
      copy_ministep(stream , "ALL_ACTIVE", ministep_name );
      add_ministep( stream , update_step , ministep_name );
      del_all_data(stream, ministep_name);
      add_data(stream,ministep_name , "SURFACE_T");
      add_data(stream,ministep_name , "SURFACE_B");
      
      del_all_obs(stream, ministep_name);
      add_obs(stream , ministep_name , "GWPR:GRANE");
      add_obs(stream , ministep_name , "GOPT:GRANE");
      add_obs(stream , ministep_name , "GOPR:GRANE");

    }




    fprintf_command( stream , INSTALL_DEFAULT_UPDATESTEP );
    fprintf(stream , "%s\n", update_step);
    fclose(stream);
  }
}
