/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'config_content_node.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#ifndef __CONFIG_CONTENT_NODE_H__
#define __CONFIG_CONTENT_NODE_H__

#ifdef __cplusplus
define extern "C" {
#endif


typedef struct config_content_node_struct config_content_node_type;

         config_content_node_type * config_content_node_alloc( const config_schema_item_type * schema );
         void                       config_content_node_set(config_content_node_type * node , const stringlist_type * token_list);
         char                     * config_content_node_alloc_joined_string(const config_content_node_type * node, const char * sep);
         void                       config_content_node_free(config_content_node_type * node);
         void                       config_content_node_free__(void * arg);
         const char               * config_content_node_iget(const config_content_node_type * node , int index);
         bool                       config_content_node_iget_as_bool(const config_content_node_type * node , int index);
         int                        config_content_node_iget_as_int(const config_content_node_type * node , int index);
         double                     config_content_node_iget_as_double(const config_content_node_type * node , double index);
         const stringlist_type    * config_content_node_get_stringlist( const config_content_node_type * node );
         void                       config_content_node_set_cwd( config_content_node_type * node , const char * config_cwd );
         const char               * config_content_node_safe_iget(const config_content_node_type * node , int index);
         int                        config_content_node_get_size( const config_content_node_type * node );
         const char * config_content_node_get_kw( const config_content_node_type * node );

#ifdef __cplusplus
}
#endif

#endif
