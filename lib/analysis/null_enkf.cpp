/*
   Copyright (C) 2011  Statoil ASA, Norway.

   The file 'null_enkf.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <ert/util/util.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/res_util/matrix_blas.hpp>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/analysis_table.hpp>
#include <ert/analysis/enkf_linalg.hpp>




void null_enkf_initX(void * module_data ,
                     matrix_type * X ,
                     const matrix_type * A ,
                     const matrix_type * S ,
                     const matrix_type * R ,
                     const matrix_type * dObs ,
                     const matrix_type * E ,
                     const matrix_type * D,
                     rng_type * rng) {

  matrix_diag_set_scalar( X , 1.0 );

}


long null_enkf_get_options( void * arg , long flag ) {
  return 0L;
}



/**
   gcc -fpic -c <object_file> -I??  <src_file>
   gcc -shared -o <lib_file> <object_files>
*/



#ifdef INTERNAL_LINK
#define LINK_NAME NULL_ENKF
#else
#define LINK_NAME EXTERNAL_MODULE_SYMBOL
#endif



analysis_table_type LINK_NAME = {
    .name            = "NULL_ENKF",
    .updateA         = NULL,
    .initX           = null_enkf_initX ,
    .init_update     = NULL,
    .complete_update = NULL,

    .freef           = NULL ,
    .alloc           = NULL ,

    .set_int         = NULL ,
    .set_double      = NULL ,
    .set_bool        = NULL ,
    .set_string      = NULL ,
    .get_options     = null_enkf_get_options,

    .has_var         = NULL,
    .get_int         = NULL,
    .get_double      = NULL,
    .get_bool        = NULL,
    .get_ptr         = NULL,
};
