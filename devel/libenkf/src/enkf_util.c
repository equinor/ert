#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <util.h>
#include <ecl_util.h>
#include <enkf_defaults.h>




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






void enkf_util_fread_assert_target_type(FILE * stream , enkf_impl_type target_type) {
  enkf_impl_type file_type;
  file_type = util_fread_int(stream);
  if (file_type != target_type) 
    util_abort("%s: wrong target type in file (expected:%d  got:%d)  - aborting \n",__func__ , target_type , file_type);

}


void enkf_util_fread_assert_target_type_from_buffer(char ** ptr , enkf_impl_type target_type) {
  enkf_impl_type file_type;
  util_fread_from_buffer( &file_type , sizeof file_type , 1 , ptr);
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



/**
  This function prints the entries in data to a file. 

  data is assumed to be num_colums pointers to double vectors of length num_rows.
  Note that this is not the conventional C style naming.

  If summarize is true, the mean and standard deviation of each column will be printed.
*/
#define PRINT_LINE(n,c,stream) { int _i; for (_i = 0; _i < (n); _i++) fputc(c , stream); fprintf(stream,"\n"); }
void enkf_util_fprintf_data(const int * index_column , const double ** data, const char * index_name , const char ** column_names, int num_rows, int num_columns, bool summarize, FILE * stream) {
  const int float_width     =  9;
  const int float_precision =  4;

  int      * width = util_malloc((num_columns + 1) * sizeof * width , __func__);
  int        total_width;

  double   * mean   = util_malloc(num_columns * sizeof * mean,   __func__);
  double   * stddev = util_malloc(num_columns * sizeof * stddev, __func__);

  /* Check the column_names. */
  for(int column_nr = 0; column_nr < num_columns; column_nr++)
  {
    if (column_names[column_nr] == NULL)
      util_abort("%s: Trying to dereference NULL pointer.\n", __func__);
  }

  /* Calculate the width of each column and the total width. */
  width[0] = strlen(index_name) + 1;
  total_width = width[0];
  for (int column_nr = 0; column_nr < num_columns; column_nr++) {
    if(column_names[column_nr] != NULL)
    width[column_nr + 1]  = util_int_max(strlen(column_names[column_nr]), 2 * float_width + 5) + 1;  /* Must accomodate A +/- B */
    width[column_nr + 1] += ( 1 - (width[column_nr + 1] & 1)); /* Ensure odd length */
    total_width += width[column_nr + 1] + 1;
  }

  /* Calculate the mean and std dev of each column. */
  for(int column_nr = 0; column_nr < num_columns; column_nr++) {
    mean  [column_nr] = util_double_vector_mean(  num_rows, data[column_nr]); 
    stddev[column_nr] = util_double_vector_stddev(num_rows, data[column_nr]); 
  }

  {

    util_fprintf_string(index_name , width[0] - 1 , true , stream); fprintf(stream , "|");
    for (int column_nr = 0; column_nr < num_columns; column_nr++) {
      util_fprintf_string(column_names[column_nr] , width[column_nr + 1] , center , stream);
      fprintf(stream , "|");
    }
    fprintf(stream , "\n");
    PRINT_LINE(total_width , '=' , stream);


    if(summarize)
    {
      util_fprintf_string("Mean" , width[0] - 1 , true , stream);
      fprintf(stream , "|");
      {
        for (int column_nr = 0; column_nr < num_columns; column_nr++) {
          int w = (width[column_nr + 1] - 5) / 2;
          util_fprintf_double(mean[column_nr] , w , float_precision , 'g' , stream);
          fprintf(stream , " +/- ");
          util_fprintf_double(stddev[column_nr] , w , float_precision , 'g' , stream);
          fprintf(stream , "|");
        }
        fprintf(stream , "\n");
      }
      PRINT_LINE(total_width , '-' , stream);
    }


    for (int row_nr = 0; row_nr < num_rows; row_nr++) {
      util_fprintf_int(index_column[row_nr], width[0] - 1 , stream);   /* This +1 is not general */
      fprintf(stream , "|");
      
      for (int column_nr = 0; column_nr < num_columns; column_nr++) {
        util_fprintf_double(data[column_nr][row_nr] , width[column_nr + 1] , float_precision , 'g' , stream);
        fprintf(stream , "|");
      }
      fprintf(stream , "\n");
    }
    PRINT_LINE(total_width , '=' , stream);
  }
  
  free(stddev);
  free(mean);
  free(width);
}
#undef PRINT_LINE


char * enkf_util_alloc_tagged_string(const char * s) {
  return util_alloc_sprintf("%s%s%s" , DEFAULT_START_TAG , s , DEFAULT_END_TAG);
}
