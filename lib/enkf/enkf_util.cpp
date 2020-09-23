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
