/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'ert_report_list.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#ifndef __ERT_REPORT_LIST_H__
#define __ERT_REPORT_LIST_H__

#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
  
  
  typedef struct ert_report_list_struct ert_report_list_type;

  ert_report_list_type  * ert_report_list_alloc();
  bool                    ert_report_list_add_report( ert_report_list_type * report_list , const char * template_path);
  void                    ert_report_list_free( ert_report_list_type * report_list );
  bool                    ert_report_list_add_path( ert_report_list_type * report_list , const char * path );
  void                    ert_report_list_set_target_path( ert_report_list_type * report_list , const char * target_path );
  void                    ert_report_list_set_plot_path( ert_report_list_type * report_list , const char * plot_path );
  int                     ert_report_list_get_num( const ert_report_list_type * report_list );
  
#ifdef __cplusplus
}
#endif
#endif
