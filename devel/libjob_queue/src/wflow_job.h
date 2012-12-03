/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'wflow_job.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __WFLOW_JOB_H__
#define __WFLOW_JOB_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <config.h>


  typedef struct wflow_job_struct wflow_job_type;
  
  const char   * wflow_job_get_name( const wflow_job_type * wflow_job );
  bool           wflow_job_internal( const wflow_job_type * wflow_job );
  config_type  * wflow_job_alloc_config();
  wflow_job_type * wflow_job_alloc(const char * name , bool internal);
  void           wflow_job_free( wflow_job_type * wflow_job );
  void           wflow_job_free__( void * arg);
  void           wflow_job_set_executable( wflow_job_type * wflow_job , const char * executable );
  wflow_job_type * wflow_job_config_alloc( const char * name , config_type * config , const char * config_file);

  void           wflow_job_update_config_compiler( const wflow_job_type * wflow_job , config_type * config_compiler );
  void           wflow_job_set_executable( wflow_job_type * wflow_job , const char * executable);
  void           wflow_job_set_function( wflow_job_type * wflow_job , const char * function);
  void           wflow_job_set_module( wflow_job_type * wflow_job , const char * module);
  
#ifdef __cplusplus
}
#endif

#endif
