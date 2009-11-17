#ifndef __ENKF_SERIALIZE_H__
#define __ENKF_SERIALIZE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdbool.h>
#include <ecl_util.h>
#include <active_list.h>
#include <matrix.h>



void enkf_matrix_serialize(const void * __node_data 	   	  , 
			   int node_size    	   	          ,      
			   ecl_type_enum node_type 	          ,           
			   const active_list_type * __active_list , 
			   matrix_type * A,
			   int row_offset, 
			   int column);


void enkf_matrix_deserialize(void * __node_data 	   	, 
			     int node_size    	   	        ,      
			     ecl_type_enum node_type 	        ,           
			     const active_list_type * __active_list , 
			     const matrix_type * A,
			     int row_offset,
			     int column);


#ifdef __cplusplus
}
#endif
#endif
