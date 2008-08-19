#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <util.h>
#include <ecl_util.h>



void enkf_util_fwrite(const void *ptr , int item_size, int items , FILE *stream , const char * caller) {
  if (fwrite(ptr , item_size , items , stream) != items) {
    fprintf(stderr,"%s: failed to write : %d bytes - aborting \n",caller , (item_size * items));
    abort();
  }
}


void enkf_util_fread(void *ptr , int item_size, int items , FILE *stream , const char * caller) {
  if (fread(ptr , item_size , items , stream) != items) {
    fprintf(stderr,"%s: failed to read : %d bytes - aborting \n",caller , (item_size * items));
    abort();
  }
}


/*****************************************************************/

static void enkf_util_rand_dbl(int N , double max , double *R) {
  int i;
  for (i=0; i < N; i++) 
    R[i] = rand() * max / RAND_MAX;
}


double enkf_util_rand_normal(double mean , double std) {
  const double pi = 3.141592653589;
  double R[2];
  enkf_util_rand_dbl(2 , 1.0 , R);
  return mean + std * sqrt(-2.0 * log(R[0])) * cos(2.0 * pi * R[1]);
}

void enkf_util_rand_stdnormal_vector(int size , double *R) {
  int i;
  for (i = 0; i < size; i++)
    R[i] = enkf_util_rand_normal(0.0 , 1.0);

}


/*****************************************************************/

#define TRUNCATE(type , void_data , size , min_ptr , max_ptr) \
{                                          \
   type * data    =   (type *) void_data;  \
   type min_value = *((type *) min_ptr);   \
   type max_value = *((type *) max_ptr);   \
   int i;                                  \
   for (i=0; i < size; i++) {              \
     if (data[i] < min_value)              \
        data[i] = min_value;               \
     else if (data[i] > max_value)         \
        data[i] = max_value;               \
   }                                       \
}  

void enkf_util_truncate(void * void_data , int size , ecl_type_enum ecl_type , void * min_ptr , void *max_ptr) {
  if (ecl_type == ecl_double_type) 
     TRUNCATE(double , void_data , size , min_ptr , max_ptr)
  else if (ecl_type == ecl_float_type)
     TRUNCATE(float , void_data , size , min_ptr , max_ptr)
  else if (ecl_type == ecl_int_type)
     TRUNCATE(int , void_data , size , min_ptr , max_ptr)
  else 
     util_abort("%s: unrecognized type - aborting \n",__func__);
}
#undef TRUNCATE



size_t enkf_util_serializeII(const void * __node_data, ecl_type_enum node_type ,  const bool * active , size_t node_offset , size_t node_size , double * serial_data , 
			     size_t serial_size , size_t serial_offset , int serial_stride ,  bool * complete) {
  
  size_t node_index;
  size_t serial_index = 0;

  if (node_type == ecl_double_type) {
    /* Serialize double -> double */
    const  double * node_data = (const double *) __node_data;
#include "serialize.h"
  } else if (node_type == ecl_float_type) {
    /* Serialize float -> double */
    const  float * node_data = (const float *) __node_data;
#include "serialize.h"
  } else 
    util_abort("%s: internal error: trying to serialize unserializable type:%s \n",__func__ , ecl_util_type_name( node_type ));

  return serial_index;

}



/*
  Denne maa ta inn type paa node_data.
*/
size_t enkf_util_serialize(const double * node_data, const bool * active , size_t node_offset , size_t node_size , double * serial_data , 
			   size_t serial_size , size_t serial_offset , int serial_stride ,  bool * complete) {
  size_t node_index;
  size_t serial_index = 0;

  if (active != NULL) {
    for (node_index = node_offset; node_index < node_size; node_index++) {
      if (active[node_index]) {
	int global_serial_index = serial_offset + serial_stride * serial_index;
	if (global_serial_index > serial_size)
	  util_abort("%s:%d fatal error global_serial_index:%d  serial_size:%d \n",__func__ , __LINE__ , global_serial_index , serial_size);
	serial_data[global_serial_index] = node_data[node_index];
	serial_index++;
	
	if (serial_offset + serial_stride * serial_index >= serial_size) {
	  if (node_index < (node_size - 1)) *complete = false;
	  break;
	}
	
      }
    }
  } else {
    for (node_index = node_offset; node_index < node_size; node_index++) {
      int global_serial_index = serial_offset + serial_stride * serial_index;
      if (global_serial_index > serial_size || global_serial_index < 0) 
	util_abort("%s:%d fatal error global_serial_index:%d  serial_size:%d \n",__func__ , __LINE__ , global_serial_index , serial_size);
      
      serial_data[global_serial_index] = node_data[node_index];
      serial_index++;
      if (serial_offset + serial_stride * serial_index >= serial_size) {
	if (node_index < (node_size - 1)) *complete = false;
	break;
      }
    }
  }

  return serial_index;
}


size_t enkf_util_deserializeII(void * __node_data , ecl_type_enum node_type , const bool * active , size_t node_offset , size_t node_size , size_t node_serial_size , 
			       const double * serial_data , size_t serial_offset , int serial_stride) {
  
  size_t serial_index = 0;
  size_t node_index;
  size_t new_node_offset = 0;
  int    last_node_index = util_int_min(node_size , node_offset + node_serial_size);
  if (last_node_index < (node_size - 1))
    new_node_offset = last_node_index;
  else
    new_node_offset = 0;
  
  if (node_type == ecl_double_type) {
    double * node_data = (double *) __node_data;
#include "deserialize.h"
  } else if (node_type == ecl_float_type) {
    float * node_data = (float *) __node_data;
#include "deserialize.h"
  } else 
    util_abort("%s: internal error: trying to deserialize unserializable type:%s \n",__func__ , ecl_util_type_name( node_type ));

  return new_node_offset;
}



size_t enkf_util_deserialize(double * node_data , const bool * active , size_t node_offset , size_t node_size , size_t node_serial_size , 
			     const double * serial_data , size_t serial_offset , int serial_stride) {
			     
  size_t serial_index = 0;
  size_t node_index;
  size_t new_node_offset = 0;
  int    last_node_index = util_int_min(node_size , node_offset + node_serial_size);
  if (last_node_index < (node_size - 1))
    new_node_offset = last_node_index;
  else
    new_node_offset = 0;
  
  if (active != NULL) {
    for (node_index = node_offset; node_index < last_node_index; node_index++) {
      if (active[node_index]) {
	node_data[node_index - node_offset] = serial_data[serial_index * serial_stride + serial_offset];
	serial_index++;
      }
    }
  }  else {
    for (node_index = node_offset; node_index < last_node_index; node_index++) {
      node_data[node_index - node_offset] = serial_data[serial_index * serial_stride + serial_offset];
      serial_index++;
    }
  }
  
  return new_node_offset;
}



void enkf_util_fread_assert_target_type(FILE * stream , enkf_impl_type target_type) {
  enkf_impl_type file_type;
  file_type = util_fread_int(stream);
  if (file_type != target_type) 
    util_abort("%s: wrong target type in file (expected:%d  got:%d)  - aborting \n",__func__ , target_type , file_type);
}


void enkf_util_fwrite_target_type(FILE * stream , enkf_impl_type target_type) {
  util_fwrite_int(target_type , stream);
}


/*
size_t util_copy_strided_vector(const void * _src, size_t src_size , int src_stride , void * _target , int target_stride , size_t target_size ,  int type_size , bool * complete) {
  const char * src    = (const char *) _src;
  char       * target = (char *)       _target;
  
  size_t src_index;
  size_t target_index = 0;

  for (src_index = 0; src_index < src_size; src_index++) {
    size_t src_adress    = src_index    * type_size * src_stride;
    size_t target_adress = target_index * type_size * target_stride;
    memcpy(&target[target_adress] , &src[src_adress] , type_size);
    target_index++;
    if (target_index == target_size) {
      if (src_index < (src_size - 1)) *complete = false;
      break;
    }
  }
  return target_index;
}

*/



/**
   Prompts the user for a filename, and reads the filename from
   stdin. 

   If the parameter 'auto_mkdir' is true, the path part of the
   filename is created automagically. If 'must_exist' is true, the
   function will loop until the user gives an existing filename.

   The filename given by the user is returned - it is the
   responibility of the calling scope to free this memory.

   The options parameter is an integer, which is a sum of the
   following alternatives:

   EXISTING_FILE  = 1
   NEW_FILE       = 2
   AUTO_MKDIR     = 4
*/

char * enkf_util_scanf_alloc_filename(const char * prompt , int options) {
  if ((options & EXISTING_FILE) && (options & NEW_FILE))
    util_abort("%s: internal error - asking for both new and existing file - impossible \n", __func__);
  {
    bool OK = true;
    char * _path;
    char file[1024];
    do {
      printf("%s",prompt);
      scanf("%s" , file);
      util_alloc_file_components(file , &_path , NULL ,  NULL);
      if (_path != NULL) {
	if (!util_path_exists(_path)) 
	  if (options & AUTO_MKDIR)
	    util_make_path(_path);
	free(_path);
      }
      
      if ((options & EXISTING_FILE) && (!util_file_exists(file)))
	OK = false;
      else if ((options & NEW_FILE) && (util_file_exists(file)))
	OK = false;

    } while (!OK);
    return util_alloc_string_copy(file);    
  }
}

