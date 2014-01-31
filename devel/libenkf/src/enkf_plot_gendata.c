/*
   Copyright (C) 2014  Statoil ASA, Norway.

   The file 'enkf_plot_gendata.c' is part of ERT - Ensemble based Reservoir Tool.

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


#include <time.h>
#include <stdbool.h>

#include <ert/util/double_vector.h>
#include <ert/util/vector.h>
#include <ert/util/thread_pool.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/block_obs.h>
#include <ert/enkf/obs_vector.h>
#include <ert/enkf/enkf_node.h>
#include <ert/enkf/enkf_node.h>

#define ENKF_PLOT_GENDATA_TYPE_ID 377626666

struct enkf_plot_gendata_struct {
  UTIL_TYPE_ID_DECLARATION;
  int size;
  const enkf_node_type * enkf_node;
  enkf_plot_genvector_type ** ensemble;
  arg_pack_type              ** work_arg;
  int                         * sort_perm;
  double_vector_type          * depth;
};

UTIL_IS_INSTANCE_FUNCTION( enkf_plot_gendata , ENKF_PLOT_GENDATA_TYPE_ID )

enkf_plot_gendata_type * enkf_plot_gendata_alloc( const enkf_node_type * enkf_node ){}


enkf_plot_gendata_type * enkf_plot_gendata_alloc( const enkf_node_type * enkf_node ){}

void enkf_plot_gendata_free( enkf_plot_gendata_type * data ){}

int  enkf_plot_gendata_get_size( const enkf_plot_gendata_type * data ){}

enkf_plot_genvector_type * enkf_plot_gendata_iget( const enkf_plot_gendata_type * plot_data , int index){}

void enkf_plot_gendata_load( enkf_plot_gendata_type * plot_data ,
                                 enkf_fs_type * fs ,
                                 int report_step ,
                                 state_enum state ,
                                 const bool_vector_type * input_mask){}

const double_vector_type * enkf_plot_gendata_get_depth( const enkf_plot_gendata_type * plot_data){}