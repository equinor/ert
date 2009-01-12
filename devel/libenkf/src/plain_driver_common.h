#ifndef __PLAIN_DRIVER_COMMON_H__
#define __PLAIN_DRIVER_COMMON_H__
#include <path_fmt.h>
#include <enkf_types.h>
#include <enkf_node.h>

void 		plain_driver_common_load_node(const char * ,  int , int , state_enum , enkf_node_type * );
void 		plain_driver_common_save_node(const char * ,  int , int , state_enum , enkf_node_type * );
path_fmt_type * plain_driver_common_realloc_path_fmt(path_fmt_type *  , const char *  , const char * , const char * );


#endif
