/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'fs_driver.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __FS_DRIVER_H__
#define __FS_DRIVER_H__ 
#ifdef __cplusplus
extern "C" {
#endif
#include <buffer.h>
#include <stringlist.h>
#include <enkf_node.h>
#include <enkf_config_node.h>


typedef struct fs_driver_struct         fs_driver_type;
typedef struct fs_index_driver_struct   fs_driver_index_type;


typedef void (save_kwlist_ftype)  (void * , int , int , buffer_type * buffer);  /* Functions used to load/store restart_kw_list instances. */
typedef void (load_kwlist_ftype)  (void * , int , int , buffer_type * buffer);          

typedef void (select_dir_ftype)   (void * driver, const char * dir , bool read , bool read_only);
typedef void (load_node_ftype)    (void * driver, const enkf_config_node_type * , int , int , buffer_type * );
typedef void (save_node_ftype)    (void * driver, const enkf_config_node_type * , int , int , buffer_type * );
typedef void (unlink_node_ftype)  (void * driver, const enkf_config_node_type * , int , int );
typedef bool (has_node_ftype)     (void * driver, const enkf_config_node_type * , int , int );
typedef void (fsync_driver_ftype) (void * driver);
typedef void (free_driver_ftype)  (void * driver);



/**
   The fs_driver_type contains a number of function pointers
   and a type_id used for run-time cast checking.
   
   The fs_driver_type is never actually used, but the point is that
   all drivers must implement the fs driver "interface". In
   practice this is done by including the macro FS_DRIVER_FIELDS
   *at the start* of the definition of another driver, i.e. the
   simplest (and only ...)  actually working driver, the plain_driver
   is implemented like this:

   struct plain_driver_struct {
      FS_DRIVER_TYPE
      int plain_driver_id;
      path_fmt_type * path;
   }


*/

   

#define FS_DRIVER_FIELDS                   \
select_dir_ftype          * select_dir;    \
load_node_ftype           * load;          \
save_node_ftype           * save;          \
has_node_ftype            * has_node;      \
unlink_node_ftype         * unlink_node;   \
free_driver_ftype         * free_driver;   \
fsync_driver_ftype        * fsync_driver;  \
int                         type_id



struct fs_driver_struct {
  FS_DRIVER_FIELDS;
  /* Fill in whatever here - i.e. dbase state. */
};




#define FS_INDEX_DRIVER_FIELDS   \
select_dir_ftype    * select_dir;   \
save_kwlist_ftype   * save_kwlist;  \
load_kwlist_ftype   * load_kwlist;  \
free_driver_ftype   * free_driver;  \
fsync_driver_ftype  * fsync_driver; \
int                   type_id;


struct fs_index_driver_struct {
  FS_INDEX_DRIVER_FIELDS;
};




/*****************************************************************/



void                       fs_driver_init(fs_driver_type * );
void                       fs_driver_assert_cast(const fs_driver_type * );
fs_driver_type   * fs_driver_safe_cast(void * );

void                       fs_driver_index_init(fs_driver_index_type * );
void                       fs_driver_index_assert_cast(const fs_driver_index_type * );
fs_driver_index_type     * fs_driver_index_safe_cast(void * );

#ifdef __cplusplus
}
#endif
#endif
