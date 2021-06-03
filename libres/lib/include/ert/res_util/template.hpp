/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'template.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_UTIL_TEMPLATE_H
#define ERT_UTIL_TEMPLATE_H
#ifdef __cplusplus
extern "C" {
#endif


#include <stdbool.h>

#include <ert/res_util/subst_list.hpp>

typedef struct template_struct template_type;


template_type * template_alloc( const char * template_file , bool internalize_template, subst_list_type * parent_subst);
void            template_free( template_type * _template );
void            template_instantiate( const template_type * _template , const char * __target_file , const subst_list_type * arg_list , bool override_symlink);
void            template_add_arg( template_type * _template , const char * key , const char * value );
subst_list_type * template_get_args_list(template_type * _template);

void            template_clear_args( template_type * _template );
int             template_add_args_from_string( template_type * _template , const char * arg_string);
char          * template_get_args_as_string( template_type * _template );
void            template_set_template_file( template_type * _template , const char * template_file);
const char    * template_get_template_file( const template_type * _template );

//void template_eval_loops( const template_type * template_ , buffer_type * buffer );


#endif
#ifdef __cplusplus
}
#endif
