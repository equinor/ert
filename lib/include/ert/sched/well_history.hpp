/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'well_history.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_WELL_HISTORY
#define ERT_WELL_HISTORY

#ifdef __cplusplus
extern "C" {
#endif
#include <ert/util/size_t_vector.hpp>

#include <ert/sched/sched_kw.hpp>
#include <ert/sched/sched_kw_wconhist.hpp>
#include <ert/sched/well_index.hpp>
#include <ert/sched/group_history.hpp>

typedef struct well_history_struct  well_history_type;




  void                  well_history_free__(void * arg);

  void                  well_history_set_parent( well_history_type * child_well , int report_step , const group_history_type * parent_group);
  group_history_type  * well_history_get_parent( well_history_type * child_well , int report_step );

  double                well_history_iget_WGPRH( const well_history_type * well_history , int report_step );
  double                well_history_iget_WOPRH( const well_history_type * well_history , int report_step );
  double                well_history_iget_WWPRH( const well_history_type * well_history , int report_step );

  UTIL_IS_INSTANCE_HEADER( well_history );


#ifdef __cplusplus
}
#endif

#endif
