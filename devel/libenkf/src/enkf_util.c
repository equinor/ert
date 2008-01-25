#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <util.h>


void * enkf_util_malloc(int byte_size , const char * caller) {
  void *tmp = malloc(byte_size);
  if (tmp == NULL) {
    fprintf(stderr,"%s: failed to allocate %d bytes - aborting \n",caller , byte_size);
    abort();
  }
  return tmp;
}

void * enkf_util_calloc(int elements , int element_size , const char * caller) {
  void *tmp = calloc(elements , element_size);
  if (tmp == NULL) {
    fprintf(stderr,"%s: failed to allocate %d bytes - aborting \n",caller , elements * element_size);
    abort();
  }
  return tmp;
}


void * enkf_util_realloc(void *ptr , int byte_size , const char * caller) {
  void *tmp = realloc(ptr , byte_size);
  if (tmp == NULL) {
    fprintf(stderr,"%s: failed to allocate %d bytes - aborting \n",caller , byte_size);
    abort();
  }
  return tmp;
}



static FILE * enkf_util_fopen(const char * filename , const char * mode , const char * text_mode , const char * caller) {
  FILE *stream = fopen(filename , mode);
  if (stream == NULL) {
    fprintf(stderr,"%s: failed to open:%s for %s - aborting \n",caller , filename , text_mode);
    abort();
  }
  return stream;
}
  
FILE * enkf_util_fopen_w(const char * filename , const char * caller) {
  return enkf_util_fopen(filename , "w" , "writing" , caller);
}

FILE * enkf_util_fopen_r(const char * filename , const char * caller) {
  return enkf_util_fopen(filename , "r" , "reading" , caller);
}


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

/*****************************************************************/



size_t enkf_util_serialize(const double * node_data, const bool * active , size_t node_offset , size_t node_size , double * serial_data , 
			   size_t serial_size , size_t serial_offset , int serial_stride ,  bool * complete) {
  size_t node_index;
  size_t serial_index = 0;

  if (active != NULL) {
    for (node_index = node_offset; node_index < node_size; node_index++) {
      if (active[node_index]) {
	serial_data[serial_offset + serial_stride * serial_index] = node_data[node_index];
	serial_index++;
	
	if (serial_offset + serial_stride * serial_index >= serial_size) {
	  if (node_index < (node_size - 1)) *complete = false;
	  break;
	}
	
      }
    }
  } else {
    for (node_index = node_offset; node_index < node_size; node_index++) {
      serial_data[serial_offset + serial_stride * serial_index] = node_data[node_index];
      serial_index++;
	
      if (serial_offset + serial_stride * serial_index >= serial_size) {
	if (node_index < (node_size - 1)) *complete = false;
	break;
      }
    }
  }
  return serial_index;
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



void enkf_util_fread_assert_target_type(FILE * stream , enkf_impl_type target_type , const char *caller) {
  enkf_impl_type file_type;
  file_type = util_fread_int(stream);
  if (file_type != target_type) {
    fprintf(stderr,"%s/%s: wrong target type in file - aborting \n",__func__ , caller);
    abort();
  }
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
