#include <sys/file.h>
#include <enkf_types.h>
#include <enkf_node.h>
#include <stdio.h>
#include <util.h>
#include <plain_driver_common.h>

/**
   This file implements small functions which are common to to 
   all the plain_driver_xxxx drivers. 
*/


/**
   The fopen calls here should probably be issued with
   util_fopen_lockf() to get exclusive access to the files - but it
   turned out be a quite massive performance hit by using lockf().
*/


void plain_driver_common_load_node(const char * filename ,  int report_step , int iens , state_enum state , enkf_node_type * node ) {
  FILE * stream   = util_fopen(filename , "r");  
  enkf_node_fread(node , stream , report_step , iens , state);
  fclose(stream);
}



void plain_driver_common_save_node(const char * filename ,  int report_step , int iens , state_enum state , enkf_node_type * node ) {
  FILE * stream = util_fopen(filename , "w");
  bool data_written = enkf_node_fwrite(node , stream , report_step , iens , state);
  fclose(stream);
  if (!data_written)
    unlink(filename);  /* The file is empty */
}
