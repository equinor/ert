/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_util.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <random>
#include <ert/util/util.h>
#include <ert/util/rng.h>
#include <ert/ecl/ecl_util.h>

#include <ert/res_util/util_printf.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/enkf_defaults.hpp>

class generator {
  rng_type *rng;

  public:
    generator(rng_type *rng): rng(rng) {}

    using value_type = unsigned int;
    static constexpr value_type min() {return 0;}
    static constexpr value_type max() {return UINT32_MAX;}

    value_type operator()() {return rng_forward(rng); }
};


double enkf_util_rand_normal(double mean , double std , rng_type * rng) {
  generator gen(rng);
  std::normal_distribution<double> normdist{mean, std};
  return normdist(gen);
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

void enkf_util_truncate(void * void_data , int size , ecl_data_type data_type , void * min_ptr , void *max_ptr) {
  if (ecl_type_is_double(data_type))
     TRUNCATE(double , void_data , size , min_ptr , max_ptr)
  else if (ecl_type_is_float(data_type))
     TRUNCATE(float , void_data , size , min_ptr , max_ptr)
  else if (ecl_type_is_int(data_type))
     TRUNCATE(int , void_data , size , min_ptr , max_ptr)
  else
     util_abort("%s: unrecognized type - aborting \n",__func__);
}
#undef TRUNCATE




void enkf_util_assert_buffer_type(buffer_type * buffer, ert_impl_type target_type) {
  ert_impl_type file_type = INVALID;
  file_type = (ert_impl_type) buffer_fread_int(buffer);
  if (file_type != target_type)
    util_abort("%s: wrong target type in file (expected:%d  got:%d) - aborting \n",
               __func__, target_type, file_type);

}


static void fread_file_type(ert_impl_type * file_type,
                            char ** buffer) {
  size_t size = sizeof *file_type;
  memcpy(file_type, *buffer, size);
  *buffer += size;
}


/**
  This function prints the entries in data to a file.

  data is assumed to be num_colums pointers to double vectors of length num_rows.
  Note that this is not the conventional C style naming.

  If summarize is true, the mean and standard deviation of each column will be printed.
*/
#define PRINT_LINE(n,c,stream) { int _i; for (_i = 0; _i < (n); _i++) fputc(c , stream); fprintf(stream,"\n"); }

void enkf_util_fprintf_data(const int * index_column ,
                            const double ** data,
                            const char * index_name ,
                            const char ** column_names,
                            int num_rows,
                            int num_columns,
                            const bool * active ,
                            bool summarize,
                            FILE * stream) {

  const int float_width     =  9;
  const int float_precision =  4;

  int * width = (int *)util_calloc((num_columns + 1) , sizeof * width );
  int        total_width;

  double * mean = (double *)util_calloc(num_columns , sizeof * mean  );
  double * stddev = (double *)util_calloc(num_columns , sizeof * stddev);

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
    if (active[column_nr]) {
      if(column_names[column_nr] != NULL)
        width[column_nr + 1]  = util_int_max(strlen(column_names[column_nr]), 2 * float_width + 5) + 1;  /* Must accomodate A +/- B */
      width[column_nr + 1] += ( 1 - (width[column_nr + 1] & 1)); /* Ensure odd length */
      total_width += width[column_nr + 1] + 1;
    }
  }

  /* Calculate the mean and std dev of each column. */
  for(int column_nr = 0; column_nr < num_columns; column_nr++) {
    if (active[column_nr]) {
      mean  [column_nr] = util_double_vector_mean(  num_rows, data[column_nr]);
      stddev[column_nr] = util_double_vector_stddev(num_rows, data[column_nr]);
    }
  }

  {
    util_fprintf_string(index_name , width[0] - 1 , left_pad, stream); fprintf(stream , "|");
    for (int column_nr = 0; column_nr < num_columns; column_nr++) {
      if (active[column_nr]) {
        util_fprintf_string(column_names[column_nr] , width[column_nr + 1] , center_pad , stream);
        fprintf(stream , "|");
      }
    }
    fprintf(stream , "\n");
    PRINT_LINE(total_width , '=' , stream);


    if(summarize)
    {
      util_fprintf_string("Mean" , width[0] - 1 ,left_pad , stream);
      fprintf(stream , "|");
      {
        for (int column_nr = 0; column_nr < num_columns; column_nr++) {
          if (active[column_nr]) {
            int w = (width[column_nr + 1] - 5) / 2;
            util_fprintf_double(mean[column_nr] , w , float_precision , 'g' , stream);
            fprintf(stream , " +/- ");
            util_fprintf_double(stddev[column_nr] , w , float_precision , 'g' , stream);
            fprintf(stream , "|");
          }
        }
        fprintf(stream , "\n");
      }
      PRINT_LINE(total_width , '-' , stream);
    }


    for (int row_nr = 0; row_nr < num_rows; row_nr++) {
      util_fprintf_int(index_column[row_nr], width[0] - 1 , stream);   /* This +1 is not general */
      fprintf(stream , "|");

      for (int column_nr = 0; column_nr < num_columns; column_nr++) {
        if (active[column_nr]) {
          util_fprintf_double(data[column_nr][row_nr] , width[column_nr + 1] , float_precision , 'g' , stream);
          fprintf(stream , "|");
        }
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

/**
   This function will compare two (key) strings. The function is
   intended to be used when sorting observation keys in summary tables
   of misfit. First the string is split on ':' - then the subsequent
   sorting is as follows:

    1. The number of items is compared, with fewer items coming first.

    2. A normal string compare is performed on the second item.

    3. A normal string compare on the first item.

    4. A normal string compare of the input key.

   The main point of this whole complexity is what is the items 2 & 3;
   this will guarantee that the different summary keys related to the
   same well, i.e. WWCT:OP1, WGOR:OP_1 and WBHP:OP_1 will come
   together.

*/


int enkf_util_compare_keys( const char * key1 , const char * key2 ) {
  int cmp;
  {
    stringlist_type * items1 = stringlist_alloc_from_split( key1 , SUMMARY_KEY_JOIN_STRING );
    stringlist_type * items2 = stringlist_alloc_from_split( key2 , SUMMARY_KEY_JOIN_STRING );

    /* 1: Compare number of items. */
    cmp = stringlist_get_size( items1 ) - stringlist_get_size( items2 );
    if (cmp == 0) {
      /* 2: String compare on second item */
      if (stringlist_get_size( items1 ) >= 2)
        cmp = strcmp( stringlist_iget( items1 , 1) , stringlist_iget( items2  , 1));
    }

    /* 3: String compare on first item */
    if (cmp == 0)
      cmp = strcmp( stringlist_iget( items1 , 0) , stringlist_iget( items2  , 0));

    /* String compare of the whole god damn thing. */
    if (cmp == 0)
      cmp = strcmp( key1 , key2 );

    stringlist_free( items2 );
    stringlist_free( items1 );
  }
  return cmp;
}


int enkf_util_compare_keys__( const void * __key1 , const void * __key2 ) {
  const char * key1 = (const char *) __key1;
  const char * key2 = (const char *) __key2;

  return enkf_util_compare_keys( key1 , key2 );
}
