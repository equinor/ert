/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'local_nodeset.h' is part of ERT - Ensemble based
   Reservoir Tool.
   
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

#ifndef __LOCAL_NODESET_H__
#define __LOCAL_NODESET_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct local_nodeset_struct local_nodeset_type;

local_nodeset_type * local_nodeset_alloc( const char * name );
void                 local_nodeset_free( local_nodeset_type * nodeset );
void                 local_nodeset_free__( void * arg );
void                 local_nodeset_add_node(local_nodeset_type * nodeset, const char *node_key);
void                 local_nodeset_del_node( local_nodeset_type * nodeset , const char * node_key);
void                 local_nodeset_clear( local_nodeset_type * nodeset);
const char *         local_nodeset_get_name( const local_nodeset_type * nodeset);
void                 local_nodeset_fprintf( const local_nodeset_type * nodeset , FILE * stream);
active_list_type   * local_nodeset_get_node_active_list(const local_nodeset_type * nodeset , const char * node_key );
stringlist_type    * local_nodeset_alloc_keys( const local_nodeset_type * nodeset );
int                  local_nodeset_get_size( const local_nodeset_type * nodeset );

#ifdef __cplusplus
}
#endif

#endif 
