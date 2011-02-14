/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'lsf_request.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __LSF_REQUEST_H__
#define __LSF_REQUEST_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <ext_joblist.h>
#include <stringlist.h>
#include <stdbool.h>

#define STATOIL_LSF_REQUEST  "select[cs && x86_64Linux]"

typedef struct lsf_request_struct lsf_request_type;


lsf_request_type * lsf_request_alloc(bool statoil_mode);
void               lsf_request_free(lsf_request_type *);
void               lsf_request_reset(lsf_request_type *);   
void               lsf_request_update(lsf_request_type * , const ext_job_type * , bool);
const char *       lsf_request_get(const lsf_request_type *);
void               lsf_request_add_manual_request(lsf_request_type *  , const char * );

#ifdef __cplusplus
}
#endif
#endif
