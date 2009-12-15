#include <block_fs.h>
#include <util.h>
#include <stringlist.h>
#include <block_fs.h>
#include <block_fs_driver.h>
#include <msg.h>



static void migrate_file( const char * src_case, int num_src_drivers , const char * target_case, int num_target_drivers, const char * file, int block_size , msg_type * msg) {
  block_fs_type ** target_fs = util_malloc( sizeof * target_fs * num_target_drivers , __func__); 
  int itarget;
  for (itarget = 0; itarget < num_target_drivers; itarget++) {
    char * path       = util_alloc_sprintf("%s/mod_%d" , target_case , itarget );
    char * mount_file = util_alloc_sprintf("%s/mod_%d/%s.mnt" , target_case , itarget , file );
    util_make_path( path );
    
    target_fs[itarget] =  block_fs_mount( mount_file , 16 , 0 , 1.0, 0 , false , false );
    free( mount_file );
    free( path );
  } 

  {
    int isrc;
    buffer_type * buffer = buffer_alloc(1024);
    for (isrc = 0; isrc < num_src_drivers; isrc++) {
      char * mount_file = util_alloc_sprintf("%s/mod_%d/%s.mnt" , src_case , isrc , file );
      block_fs_type   * src_fs    = block_fs_mount( mount_file , 16 , 1024 , 1.0 , 0 , true , true );
      vector_type     * file_list = block_fs_alloc_filelist( src_fs , NULL , NO_SORT , false );
      int ifile;
      msg_update( msg , mount_file );
      for (ifile = 0; ifile < vector_get_size( file_list ); ifile++) {
        const file_node_type * node = vector_iget_const( file_list , ifile );
        const char * filename       = file_node_get_filename( node );
        int   report_step , iens;
        char * key;
        if (block_fs_sscanf_key( filename , &key , &report_step , &iens )) {
          block_fs_fread_realloc_buffer( src_fs , filename , buffer);
          block_fs_fwrite_buffer( target_fs[(iens % num_target_drivers)] , filename , buffer );
          free( key );
        } else
          util_abort("%s: All hell is loose - failed to parse:%s \n",__func__ , filename);
      }

      vector_free( file_list );
      block_fs_close(src_fs , false);
    }
    buffer_free( buffer );
  }

  
  for (itarget = 0; itarget < num_target_drivers; itarget++) 
    block_fs_close( target_fs[itarget] , false);
  free( target_fs );
}


static void copy_index( const char * src_case , const char * target_case) {
  char * mount_src    = util_alloc_filename(src_case , "INDEX" , "mnt");
  char * mount_target = util_alloc_filename(target_case , "INDEX" , "mnt");
  char * data_src     = util_alloc_filename(src_case , "INDEX" , "data_0");
  char * data_target  = util_alloc_filename(target_case , "INDEX" , "data_0");

  util_copy_file( mount_src , mount_target );
  util_copy_file( data_src , data_target );

  free( mount_src );
  free( mount_target );
  free( data_src );
  free( data_target );
}


static void usage() {
  printf("Use:\n");
  printf("bash%% migrate_bfs <Source ENSPATH>  <Target ENSPATH> case\n");
  exit(1);
}

int main(int argc, char ** argv) {
  int num_src_drivers    = 10;
  int num_target_drivers = 32;
  if (argc != 4) 
    usage();
  
  {
    char * src_path        = argv[1] ;
    char * target_path     = argv[2] ;
    char * dir             = argv[3] ;
   
    util_make_path( target_path );
    if (util_same_file( src_path , target_path)) {
      fprintf(stderr,"The two directories:%s and %s point to the same location \n" , src_path , target_path );
      exit(1);
    }
    
    {
      char * src_case    = util_alloc_sprintf("%s/%s" , src_path , dir );
      char * target_case = util_alloc_sprintf("%s/%s" , target_path , dir );

      msg_type * msg = msg_alloc("Copying from: ");
      msg_show( msg );
      migrate_file(src_case , num_src_drivers , target_case , num_target_drivers , "ANALYZED" , 32 , msg);
      migrate_file(src_case , num_src_drivers , target_case , num_target_drivers , "FORECAST" , 32 , msg);
      migrate_file(src_case , num_src_drivers , target_case , num_target_drivers , "PARAMETER" , 32 , msg);
      migrate_file(src_case , num_src_drivers , target_case , num_target_drivers , "STATIC" , 32 , msg);
      copy_index( src_case , target_case );
      free( src_case);
      free( target_case );
      msg_free( msg , true );
    }
  }
}
